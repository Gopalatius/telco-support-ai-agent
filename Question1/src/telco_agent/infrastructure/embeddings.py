from typing import cast

from google import genai
from google.genai import types

from telco_agent.domain.models import KnowledgeChunk
from telco_agent.settings import Settings


class GeminiEmbeddingClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = genai.Client(api_key=settings.gemini_api_key)

    def embed_query(self, text: str) -> list[float]:
        response = self._client.models.embed_content(
            model=self._settings.embedding_model,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=3072,
            ),
        )
        embeddings = cast("list[object]", response.embeddings or [])
        if not embeddings:
            raise RuntimeError("Gemini did not return a query embedding.")
        values = cast("list[float] | None", getattr(embeddings[0], "values", None))
        if values is None:
            raise RuntimeError("Gemini returned an empty query embedding payload.")
        return list(values)

    def embed_documents(self, chunks: list[KnowledgeChunk]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for chunk in chunks:
            response = self._client.models.embed_content(
                model=self._settings.embedding_model,
                contents=chunk.embedding_text,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    title=chunk.title,
                    output_dimensionality=3072,
                ),
            )
            embeddings = cast("list[object]", response.embeddings or [])
            if len(embeddings) != 1:
                raise RuntimeError(
                    "Gemini returned an unexpected number of document embeddings."
                )
            embedding = embeddings[0]
            values = cast("list[float] | None", getattr(embedding, "values", None))
            if values is None:
                raise RuntimeError(
                    "Gemini returned an empty document embedding payload."
                )
            vectors.append(list(values))
        return vectors
