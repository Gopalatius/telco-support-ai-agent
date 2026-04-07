from typing import Any, cast

from openai import OpenAI
from pydantic import ValidationError

from telco_agent.domain.knowledge_base import format_query_for_embedding
from telco_agent.domain.models import (
    ChatMessage,
    GeneratorClient,
    KnowledgeChunk,
    LlmDecision,
    RetrievalQueryPlan,
)
from telco_agent.domain.prompting import (
    RETRIEVAL_QUERY_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_history,
    build_retrieval_query_prompt,
    build_user_prompt,
)
from telco_agent.settings import Settings


class OpenRouterGeneratorClient(GeneratorClient):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = OpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
        )

    def compose_retrieval_query(
        self,
        *,
        message: str,
        history: list[ChatMessage],
    ) -> RetrievalQueryPlan:
        schema = {
            "type": "json_schema",
            "name": "retrieval_query_plan",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "search_query": {"type": "string"},
                    "conversation_focus": {"type": "string"},
                },
                "required": ["search_query", "conversation_focus"],
                "additionalProperties": False,
            },
        }

        responses_api = cast("Any", self._client.responses)
        response = self._create_structured_response(
            responses_api=responses_api,
            model=self._settings.openrouter_query_rewrite_model,
            input_items=[
                {"role": "system", "content": RETRIEVAL_QUERY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_retrieval_query_prompt(
                        message=message,
                        history=history,
                        max_turns=self._settings.max_history_turns,
                    ),
                },
            ],
            schema=schema,
            provider_preferences=self._build_provider_preferences(
                order=self._settings.openrouter_query_rewrite_provider_order_list,
                allow_fallbacks=self._settings.openrouter_query_rewrite_allow_fallbacks,
                require_parameters=self._settings.openrouter_query_rewrite_require_parameters,
            ),
        )
        payload = cast("str | None", getattr(response, "output_text", None))
        if not payload:
            return RetrievalQueryPlan(
                search_query=format_query_for_embedding(message),
                conversation_focus=message,
            )

        try:
            plan = RetrievalQueryPlan.model_validate_json(payload)
        except ValidationError:
            return RetrievalQueryPlan(
                search_query=format_query_for_embedding(message),
                conversation_focus=message,
            )

        cleaned_query = " ".join(plan.search_query.split()).strip()
        if not self._is_useful_search_query(cleaned_query):
            return RetrievalQueryPlan(
                search_query=format_query_for_embedding(message),
                conversation_focus=message,
            )

        return RetrievalQueryPlan(
            search_query=format_query_for_embedding(cleaned_query),
            conversation_focus=plan.conversation_focus,
        )

    def generate_reply(
        self,
        *,
        message: str,
        history: list[ChatMessage],
        retrieved_chunks: list[KnowledgeChunk],
    ) -> LlmDecision:
        input_items: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        input_items.extend(
            build_history(history, max_turns=self._settings.max_history_turns)
        )
        input_items.append(
            {
                "role": "user",
                "content": build_user_prompt(message=message, chunks=retrieved_chunks),
            }
        )

        schema = {
            "type": "json_schema",
            "name": "chat_decision",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "reply": {"type": "string"},
                    "escalate": {"type": "boolean"},
                    "confidence": {"type": "number"},
                    "rationale": {"type": "string"},
                },
                "required": ["reply", "escalate", "confidence", "rationale"],
                "additionalProperties": False,
            },
        }

        responses_api = cast("Any", self._client.responses)
        response = self._create_structured_response(
            responses_api=responses_api,
            model=self._settings.openrouter_model,
            input_items=input_items,
            schema=schema,
            provider_preferences=self._build_provider_preferences(
                order=self._settings.openrouter_provider_order_list,
                allow_fallbacks=self._settings.openrouter_provider_allow_fallbacks,
                require_parameters=self._settings.openrouter_provider_require_parameters,
            ),
        )

        payload = cast("str | None", getattr(response, "output_text", None))
        if not payload:
            raise RuntimeError("OpenRouter did not return output_text.")

        try:
            return LlmDecision.model_validate_json(payload)
        except ValidationError as error:
            raise RuntimeError("Model response failed schema validation.") from error

    @staticmethod
    def _is_useful_search_query(query: str) -> bool:
        alnum_tokens = [
            token
            for token in query.split()
            if any(character.isalnum() for character in token)
        ]
        return len(alnum_tokens) >= 2

    @staticmethod
    def _build_provider_preferences(
        *,
        order: list[str] | None,
        allow_fallbacks: bool,
        require_parameters: bool,
    ) -> dict[str, Any] | None:
        if not order:
            return None

        return {
            "order": order,
            "allow_fallbacks": allow_fallbacks,
            "require_parameters": require_parameters,
        }

    @staticmethod
    def _create_structured_response(
        *,
        responses_api: Any,
        model: str,
        input_items: list[dict[str, str]],
        schema: dict[str, Any],
        provider_preferences: dict[str, Any] | None,
    ) -> Any:
        request_kwargs: dict[str, Any] = {
            "model": model,
            "input": input_items,
            "text": {"format": schema},
        }
        if provider_preferences is not None:
            request_kwargs["extra_body"] = {"provider": provider_preferences}
        return responses_api.create(**request_kwargs)
