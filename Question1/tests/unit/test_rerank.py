import io
from urllib.error import HTTPError, URLError

import pytest

from telco_agent.domain.models import KnowledgeChunk
from telco_agent.infrastructure.rerank import OpenRouterReranker
from telco_agent.settings import Settings


class StubHttpResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


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


def test_rerank_returns_ranked_chunks_and_tracks_usage(monkeypatch) -> None:
    monkeypatch.setattr(
        "telco_agent.infrastructure.rerank.request.urlopen",
        lambda request, timeout: StubHttpResponse(
            b"".join(
                [
                    b'{"results":[{"index":1,"relevance_score":0.91},{"index":0,"relevance_score":0.72}],',
                    b'"usage":{"search_units":2}}',
                ]
            )
        ),
    )
    reranker = OpenRouterReranker(make_settings())

    result = reranker.rerank(
        query_text="late fee",
        chunks=[
            KnowledgeChunk(
                "billing-1", "Billing", "billing.md", "Late fee", "embedding"
            ),
            KnowledgeChunk(
                "billing-2", "Billing", "billing.md", "Dispute window", "embedding"
            ),
        ],
        top_n=2,
    )

    assert [chunk.chunk_id for chunk in result.chunks] == ["billing-2", "billing-1"]
    assert result.chunks[0].metadata["reranked"] == "true"
    assert reranker.last_search_units == 2


def test_rerank_returns_empty_for_no_chunks() -> None:
    reranker = OpenRouterReranker(make_settings())

    assert reranker.rerank(query_text="late fee", chunks=[], top_n=3).chunks == []


def test_rerank_raises_for_http_error(monkeypatch) -> None:
    def raising_urlopen(request, timeout):
        raise HTTPError(
            url="https://openrouter.ai/api/v1/rerank",
            code=400,
            msg="bad",
            hdrs=None,
            fp=io.BytesIO(b'{"error":"bad request"}'),
        )

    monkeypatch.setattr(
        "telco_agent.infrastructure.rerank.request.urlopen", raising_urlopen
    )
    reranker = OpenRouterReranker(make_settings())

    with pytest.raises(RuntimeError, match="bad request"):
        reranker.rerank(
            query_text="late fee",
            chunks=[
                KnowledgeChunk(
                    "billing-1", "Billing", "billing.md", "Late fee", "embedding"
                )
            ],
            top_n=1,
        )


def test_rerank_raises_for_url_error(monkeypatch) -> None:
    monkeypatch.setattr(
        "telco_agent.infrastructure.rerank.request.urlopen",
        lambda request, timeout: (_ for _ in ()).throw(URLError("offline")),
    )
    reranker = OpenRouterReranker(make_settings())

    with pytest.raises(RuntimeError, match="could not reach the endpoint"):
        reranker.rerank(
            query_text="late fee",
            chunks=[
                KnowledgeChunk(
                    "billing-1", "Billing", "billing.md", "Late fee", "embedding"
                )
            ],
            top_n=1,
        )
