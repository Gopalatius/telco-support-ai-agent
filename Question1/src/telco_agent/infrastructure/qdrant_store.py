from typing import Any, cast
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.http import models

from telco_agent.domain.models import KnowledgeChunk, RetrievalResult
from telco_agent.settings import Settings


class QdrantKnowledgeStore:
    def __init__(self, settings: Settings, client: QdrantClient | None = None) -> None:
        self._settings = settings
        self._client = client or QdrantClient(url=settings.qdrant_url)

    def recreate_collection(self) -> None:
        if self._client.collection_exists(self._settings.qdrant_collection):
            self._client.delete_collection(self._settings.qdrant_collection)
        self._client.create_collection(
            collection_name=self._settings.qdrant_collection,
            vectors_config=models.VectorParams(
                size=3072, distance=models.Distance.COSINE
            ),
        )

    def upsert_chunks(
        self, chunks: list[KnowledgeChunk], vectors: list[list[float]]
    ) -> None:
        points = [
            models.PointStruct(
                id=str(uuid5(NAMESPACE_URL, chunk.chunk_id)),
                vector=vector,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "title": chunk.title,
                    "source": chunk.source,
                    "text": chunk.text,
                    **chunk.metadata,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self._client.upsert(
            collection_name=self._settings.qdrant_collection, points=points
        )

    def search(
        self, vector: list[float], limit: int, min_score: float
    ) -> RetrievalResult:
        result = self._client.query_points(
            collection_name=self._settings.qdrant_collection,
            query=vector,
            limit=limit,
            with_payload=True,
            score_threshold=min_score,
        )
        hits = cast("list[Any]", result.points)

        chunks = [
            KnowledgeChunk(
                chunk_id=str(hit.payload["chunk_id"]),
                title=str(hit.payload["title"]),
                source=str(hit.payload["source"]),
                text=str(hit.payload["text"]),
                embedding_text="",
                metadata={
                    key: str(value)
                    for key, value in hit.payload.items()
                    if key not in {"chunk_id", "title", "source", "text"}
                },
                score=hit.score,
            )
            for hit in hits
        ]
        return RetrievalResult(chunks=chunks)

    def healthcheck(self) -> bool:
        self._client.get_collections()
        return True
