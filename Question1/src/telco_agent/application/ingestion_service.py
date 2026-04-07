from pathlib import Path

from telco_agent.domain.knowledge_base import load_knowledge_chunks
from telco_agent.domain.models import EmbeddingClient, VectorStore


class IngestionService:
    def __init__(
        self,
        *,
        embedding_client: EmbeddingClient,
        vector_store: VectorStore,
    ) -> None:
        self._embedding_client = embedding_client
        self._vector_store = vector_store

    def ingest(self, knowledge_base_dir: Path) -> int:
        chunks = load_knowledge_chunks(knowledge_base_dir=knowledge_base_dir)
        vectors = self._embedding_client.embed_documents(chunks)
        self._vector_store.recreate_collection()
        self._vector_store.upsert_chunks(chunks, vectors)
        return len(chunks)
