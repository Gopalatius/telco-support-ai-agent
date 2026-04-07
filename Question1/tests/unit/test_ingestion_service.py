from pathlib import Path

from telco_agent.application.ingestion_service import IngestionService
from telco_agent.domain.models import KnowledgeChunk


class StubEmbeddingClient:
    def __init__(self) -> None:
        self.chunks: list[KnowledgeChunk] | None = None

    def embed_documents(self, chunks: list[KnowledgeChunk]) -> list[list[float]]:
        self.chunks = chunks
        return [[0.1, 0.2] for _ in chunks]


class StubVectorStore:
    def __init__(self) -> None:
        self.recreated = False
        self.upserted_chunks: list[KnowledgeChunk] = []
        self.upserted_vectors: list[list[float]] = []

    def recreate_collection(self) -> None:
        self.recreated = True

    def upsert_chunks(
        self, chunks: list[KnowledgeChunk], vectors: list[list[float]]
    ) -> None:
        self.upserted_chunks = chunks
        self.upserted_vectors = vectors


def test_ingestion_service_loads_embeds_and_upserts(monkeypatch) -> None:
    chunks = [
        KnowledgeChunk(
            "billing-1", "Billing Policy", "billing.md", "Late fee", "embedding"
        )
    ]
    monkeypatch.setattr(
        "telco_agent.application.ingestion_service.load_knowledge_chunks",
        lambda knowledge_base_dir: chunks if knowledge_base_dir == Path("/kb") else [],
    )

    embedding_client = StubEmbeddingClient()
    vector_store = StubVectorStore()
    service = IngestionService(
        embedding_client=embedding_client, vector_store=vector_store
    )

    count = service.ingest(Path("/kb"))

    assert count == 1
    assert embedding_client.chunks == chunks
    assert vector_store.recreated is True
    assert vector_store.upserted_chunks == chunks
    assert vector_store.upserted_vectors == [[0.1, 0.2]]
