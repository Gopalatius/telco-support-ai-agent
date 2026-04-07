from types import SimpleNamespace

import pytest

from telco_agent.domain.models import KnowledgeChunk
from telco_agent.infrastructure.embeddings import GeminiEmbeddingClient
from telco_agent.settings import Settings


class StubModels:
    def __init__(self, responses) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def embed_content(self, *, model: str, contents, config) -> object:
        self.calls.append({"model": model, "contents": contents, "config": config})
        return self._responses.pop(0)


def make_settings() -> Settings:
    return Settings.model_construct(
        openrouter_api_key="test",
        gemini_api_key="gem-key",
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


def test_embed_query_and_documents_return_vectors(monkeypatch) -> None:
    responses = [
        SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1, 0.2])]),
        SimpleNamespace(embeddings=[SimpleNamespace(values=[1.0, 0.0])]),
        SimpleNamespace(embeddings=[SimpleNamespace(values=[0.0, 1.0])]),
    ]
    models = StubModels(responses)
    monkeypatch.setattr(
        "telco_agent.infrastructure.embeddings.genai.Client",
        lambda api_key: SimpleNamespace(models=models, api_key=api_key),
    )

    client = GeminiEmbeddingClient(make_settings())

    assert client.embed_query("late fee") == [0.1, 0.2]
    assert client.embed_documents(
        [
            KnowledgeChunk("a", "Title", "source", "text", "embedding"),
            KnowledgeChunk("b", "Title", "source", "text", "embedding"),
        ]
    ) == [[1.0, 0.0], [0.0, 1.0]]
    assert len(models.calls) == 3
    assert models.calls[0]["model"] == "gemini-embedding-2-preview"
    assert models.calls[0]["config"].task_type == "RETRIEVAL_QUERY"
    assert models.calls[1]["config"].task_type == "RETRIEVAL_DOCUMENT"
    assert models.calls[1]["config"].title == "Title"


def test_embed_query_raises_for_missing_payload(monkeypatch) -> None:
    monkeypatch.setattr(
        "telco_agent.infrastructure.embeddings.genai.Client",
        lambda api_key: SimpleNamespace(
            models=StubModels([SimpleNamespace(embeddings=[])]),
            api_key=api_key,
        ),
    )

    client = GeminiEmbeddingClient(make_settings())

    with pytest.raises(RuntimeError, match="did not return a query embedding"):
        client.embed_query("late fee")


def test_embed_query_raises_for_missing_values(monkeypatch) -> None:
    monkeypatch.setattr(
        "telco_agent.infrastructure.embeddings.genai.Client",
        lambda api_key: SimpleNamespace(
            models=StubModels(
                [SimpleNamespace(embeddings=[SimpleNamespace(values=None)])]
            ),
            api_key=api_key,
        ),
    )

    client = GeminiEmbeddingClient(make_settings())

    with pytest.raises(RuntimeError, match="empty query embedding payload"):
        client.embed_query("late fee")


def test_embed_documents_raises_for_unexpected_count(monkeypatch) -> None:
    monkeypatch.setattr(
        "telco_agent.infrastructure.embeddings.genai.Client",
        lambda api_key: SimpleNamespace(
            models=StubModels(
                [
                    SimpleNamespace(
                        embeddings=[
                            SimpleNamespace(values=[1.0, 0.0]),
                            SimpleNamespace(values=[0.0, 1.0]),
                        ]
                    )
                ]
            ),
            api_key=api_key,
        ),
    )

    client = GeminiEmbeddingClient(make_settings())

    with pytest.raises(RuntimeError, match="unexpected number of document embeddings"):
        client.embed_documents(
            [
                KnowledgeChunk("a", "Title", "source", "text", "embedding"),
                KnowledgeChunk("b", "Title", "source", "text", "embedding"),
            ]
        )


def test_embed_documents_raises_for_missing_values(monkeypatch) -> None:
    monkeypatch.setattr(
        "telco_agent.infrastructure.embeddings.genai.Client",
        lambda api_key: SimpleNamespace(
            models=StubModels(
                [SimpleNamespace(embeddings=[SimpleNamespace(values=None)])]
            ),
            api_key=api_key,
        ),
    )

    client = GeminiEmbeddingClient(make_settings())

    with pytest.raises(RuntimeError, match="empty document embedding payload"):
        client.embed_documents(
            [KnowledgeChunk("a", "Title", "source", "text", "embedding")]
        )
