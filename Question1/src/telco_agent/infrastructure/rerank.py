import json
from typing import Any, cast
from urllib import request
from urllib.error import HTTPError, URLError

from telco_agent.domain.models import KnowledgeChunk, Reranker, RetrievalResult
from telco_agent.settings import Settings


class OpenRouterReranker(Reranker):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._endpoint = f"{settings.openrouter_base_url.rstrip('/')}/rerank"
        self.last_search_units = 0

    def rerank(
        self,
        *,
        query_text: str,
        chunks: list[KnowledgeChunk],
        top_n: int,
    ) -> RetrievalResult:
        if not chunks:
            return RetrievalResult(chunks=[])

        payload = json.dumps(
            {
                "model": self._settings.rerank_model,
                "query": query_text,
                "documents": [
                    f"title: {chunk.title} | text: {chunk.text}" for chunk in chunks
                ],
                "top_n": min(top_n, len(chunks)),
            }
        ).encode("utf-8")

        http_request = request.Request(
            self._endpoint,
            method="POST",
            data=payload,
            headers={
                "Authorization": f"Bearer {self._settings.openrouter_api_key}",
                "Content-Type": "application/json",
            },
        )

        try:
            with request.urlopen(http_request, timeout=30) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"OpenRouter rerank request failed: {detail}") from error
        except URLError as error:
            raise RuntimeError(
                "OpenRouter rerank request could not reach the endpoint."
            ) from error

        usage = cast("dict[str, Any]", response_payload.get("usage", {}))
        self.last_search_units = int(usage.get("search_units", 0) or 0)
        results = cast("list[dict[str, Any]]", response_payload.get("results", []))
        reranked_chunks: list[KnowledgeChunk] = []

        for result in results:
            index = cast("int | None", result.get("index"))
            if index is None or index < 0 or index >= len(chunks):
                continue
            source_chunk = chunks[index]
            reranked_chunks.append(
                KnowledgeChunk(
                    chunk_id=source_chunk.chunk_id,
                    title=source_chunk.title,
                    source=source_chunk.source,
                    text=source_chunk.text,
                    embedding_text=source_chunk.embedding_text,
                    metadata={
                        **source_chunk.metadata,
                        "reranked": "true",
                    },
                    score=cast("float | None", result.get("relevance_score")),
                )
            )

        return RetrievalResult(chunks=reranked_chunks)
