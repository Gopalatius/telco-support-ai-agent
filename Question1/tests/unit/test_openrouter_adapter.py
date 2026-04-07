from types import SimpleNamespace

import pytest

from telco_agent.domain.models import ChatMessage, KnowledgeChunk
from telco_agent.infrastructure.openrouter import OpenRouterGeneratorClient
from telco_agent.settings import Settings


class StubResponsesApi:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def make_settings() -> Settings:
    return Settings.model_construct(
        openrouter_api_key="test",
        gemini_api_key="test",
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_model="moonshotai/kimi-k2.5",
        openrouter_provider_order="moonshotai/int4",
        openrouter_provider_allow_fallbacks=True,
        openrouter_provider_require_parameters=True,
        openrouter_query_rewrite_model="openai/gpt-oss-20b",
        openrouter_query_rewrite_provider_order="groq",
        openrouter_query_rewrite_allow_fallbacks=True,
        openrouter_query_rewrite_require_parameters=True,
        embedding_model="gemini-embedding-2-preview",
        qdrant_url="http://localhost:6333",
        qdrant_collection="telco_knowledge_base",
        retrieval_top_k=3,
        retrieval_min_score=0.65,
        retrieval_strategy="dense",
        retrieval_candidate_pool=6,
        retrieval_fusion_k=60,
        rerank_enabled=False,
        rerank_model="cohere/rerank-4-fast",
        rerank_top_n=3,
        rerank_candidate_pool=8,
        max_history_turns=6,
        app_name="test",
    )


def build_client(stub_responses: StubResponsesApi) -> OpenRouterGeneratorClient:
    client = OpenRouterGeneratorClient(make_settings())
    client._client = SimpleNamespace(responses=stub_responses)
    return client


def test_query_sanity_check_rejects_punctuation_only_search_queries() -> None:
    assert OpenRouterGeneratorClient._is_useful_search_query(",") is False
    assert (
        OpenRouterGeneratorClient._is_useful_search_query("billing dispute deadline")
        is True
    )


def test_build_provider_preferences_returns_none_without_order() -> None:
    assert (
        OpenRouterGeneratorClient._build_provider_preferences(
            order=None,
            allow_fallbacks=True,
            require_parameters=True,
        )
        is None
    )


def test_build_provider_preferences_keeps_order_and_flags() -> None:
    assert OpenRouterGeneratorClient._build_provider_preferences(
        order=["groq"],
        allow_fallbacks=False,
        require_parameters=True,
    ) == {
        "order": ["groq"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }


def test_compose_retrieval_query_uses_rewrite_model_and_provider_preferences() -> None:
    responses = StubResponsesApi(
        [
            SimpleNamespace(
                output_text=(
                    '{"search_query":"billing dispute deadline",'
                    '"conversation_focus":"dispute question"}'
                )
            )
        ]
    )
    client = build_client(responses)

    plan = client.compose_retrieval_query(
        message="Can I still dispute it?",
        history=[ChatMessage(role="user", content="My invoice is wrong.")],
    )

    assert (
        plan.search_query
        == "task: question answering | query: billing dispute deadline"
    )
    assert plan.conversation_focus == "dispute question"
    assert responses.calls[0]["model"] == "openai/gpt-oss-20b"
    assert responses.calls[0]["extra_body"] == {
        "provider": {
            "order": ["groq"],
            "allow_fallbacks": True,
            "require_parameters": True,
        }
    }


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (None, "fallback please"),
        ("not json", "fallback please"),
        ('{"search_query":",","conversation_focus":"focus"}', "fallback please"),
    ],
)
def test_compose_retrieval_query_falls_back_when_payload_is_unusable(
    payload, message: str
) -> None:
    responses = StubResponsesApi([SimpleNamespace(output_text=payload)])
    client = build_client(responses)

    plan = client.compose_retrieval_query(message=message, history=[])

    assert plan.search_query == f"task: question answering | query: {message}"
    assert plan.conversation_focus == message


def test_generate_reply_returns_valid_decision_and_main_provider_preferences() -> None:
    responses = StubResponsesApi(
        [
            SimpleNamespace(
                output_text=(
                    '{"reply":"Late fee is IDR 50,000.","escalate":false,'
                    '"confidence":0.9,"rationale":"grounded"}'
                )
            )
        ]
    )
    client = build_client(responses)

    decision = client.generate_reply(
        message="What is the late fee?",
        history=[],
        retrieved_chunks=[
            KnowledgeChunk(
                "billing-1",
                "Billing Policy",
                "billing_policy.md",
                "Late fee is IDR 50,000 after 14 days overdue",
                "embedding",
            )
        ],
    )

    assert decision.reply == "Late fee is IDR 50,000."
    assert decision.escalate is False
    assert responses.calls[0]["model"] == "moonshotai/kimi-k2.5"
    assert responses.calls[0]["extra_body"] == {
        "provider": {
            "order": ["moonshotai/int4"],
            "allow_fallbacks": True,
            "require_parameters": True,
        }
    }


def test_generate_reply_raises_when_output_text_is_missing() -> None:
    client = build_client(StubResponsesApi([SimpleNamespace(output_text=None)]))

    with pytest.raises(RuntimeError, match="did not return output_text"):
        client.generate_reply(message="late fee", history=[], retrieved_chunks=[])


def test_generate_reply_raises_when_schema_validation_fails() -> None:
    client = build_client(
        StubResponsesApi([SimpleNamespace(output_text='{"reply":"x"}')])
    )

    with pytest.raises(RuntimeError, match="schema validation"):
        client.generate_reply(message="late fee", history=[], retrieved_chunks=[])
