from qdrant_client import QdrantClient

from telco_agent.domain.models import KnowledgeChunk
from telco_agent.infrastructure.qdrant_store import QdrantKnowledgeStore
from telco_agent.settings import Settings


def build_settings() -> Settings:
    return Settings.model_construct(
        openrouter_api_key="test",
        gemini_api_key="test",
        qdrant_collection="test_collection",
        qdrant_url="http://localhost:6333",
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


def test_qdrant_search_returns_relevant_chunk() -> None:
    store = QdrantKnowledgeStore(
        settings=build_settings(),
        client=QdrantClient(location=":memory:"),
    )
    store.recreate_collection()
    store.upsert_chunks(
        chunks=[
            KnowledgeChunk(
                chunk_id="billing-1",
                title="Billing Policy",
                source="billing_policy.md",
                text="Late payment fee of IDR 50,000 applies after 14 days overdue",
                embedding_text="",
            ),
            KnowledgeChunk(
                chunk_id="plan-1",
                title="Service Plans",
                source="service_plans.md",
                text="Unlimited Plan: IDR 299,000/month",
                embedding_text="",
            ),
        ],
        vectors=[
            [1.0] + [0.0] * 3071,
            [0.0, 1.0] + [0.0] * 3070,
        ],
    )

    result = store.search([1.0] + [0.0] * 3071, limit=1, min_score=0.1)

    assert len(result.chunks) == 1
    assert result.chunks[0].chunk_id == "billing-1"


def test_qdrant_store_builds_default_client_and_healthchecks(monkeypatch) -> None:
    class StubClient:
        def __init__(self, *, url: str) -> None:
            self.url = url
            self.checked = False

        def get_collections(self) -> None:
            self.checked = True

    created: list[StubClient] = []

    def stub_qdrant_client(*, url: str) -> StubClient:
        client = StubClient(url=url)
        created.append(client)
        return client

    monkeypatch.setattr(
        "telco_agent.infrastructure.qdrant_store.QdrantClient", stub_qdrant_client
    )

    store = QdrantKnowledgeStore(settings=build_settings())

    assert store.healthcheck() is True
    assert created[0].url == "http://localhost:6333"
    assert created[0].checked is True
