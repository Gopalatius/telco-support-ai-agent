from collections import defaultdict
from re import findall

from rank_bm25 import BM25Okapi

from telco_agent.domain.models import KnowledgeChunk, RetrievalResult, VectorStore


def normalize_for_lexical_search(text: str) -> list[str]:
    return findall(r"[a-z0-9]+", text.lower())


class DenseRetriever:
    def __init__(self, vector_store: VectorStore) -> None:
        self._vector_store = vector_store

    def search(
        self,
        *,
        query_text: str,
        query_vector: list[float],
        limit: int,
        min_score: float,
    ) -> RetrievalResult:
        del query_text
        return self._vector_store.search(query_vector, limit=limit, min_score=min_score)


class BM25Retriever:
    def __init__(self, chunks: list[KnowledgeChunk]) -> None:
        self._chunks = chunks
        self._tokenized_chunks = [
            normalize_for_lexical_search(f"{chunk.title} {chunk.text}")
            for chunk in chunks
        ]
        self._bm25 = BM25Okapi(self._tokenized_chunks)

    def search(
        self,
        *,
        query_text: str,
        query_vector: list[float],
        limit: int,
        min_score: float,
    ) -> RetrievalResult:
        del query_vector, min_score
        tokenized_query = normalize_for_lexical_search(query_text)
        if not tokenized_query:
            return RetrievalResult(chunks=[])

        query_terms = set(tokenized_query)
        scores = self._bm25.get_scores(tokenized_query)
        ranked_items = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)

        chunks: list[KnowledgeChunk] = []
        for index, score in ranked_items:
            if len(chunks) >= limit:
                break
            if not query_terms.intersection(self._tokenized_chunks[index]):
                continue

            source_chunk = self._chunks[index]
            chunks.append(
                KnowledgeChunk(
                    chunk_id=source_chunk.chunk_id,
                    title=source_chunk.title,
                    source=source_chunk.source,
                    text=source_chunk.text,
                    embedding_text=source_chunk.embedding_text,
                    metadata=dict(source_chunk.metadata),
                    score=float(score),
                )
            )

        return RetrievalResult(chunks=chunks)


class HybridRetriever:
    def __init__(
        self,
        *,
        dense_retriever: DenseRetriever,
        lexical_retriever: BM25Retriever,
        candidate_pool: int,
        fusion_k: int,
    ) -> None:
        self._dense_retriever = dense_retriever
        self._lexical_retriever = lexical_retriever
        self._candidate_pool = candidate_pool
        self._fusion_k = fusion_k

    def search(
        self,
        *,
        query_text: str,
        query_vector: list[float],
        limit: int,
        min_score: float,
    ) -> RetrievalResult:
        dense_result = self._dense_retriever.search(
            query_text=query_text,
            query_vector=query_vector,
            limit=max(limit, self._candidate_pool),
            min_score=min_score,
        )
        lexical_result = self._lexical_retriever.search(
            query_text=query_text,
            query_vector=query_vector,
            limit=max(limit, self._candidate_pool),
            min_score=min_score,
        )

        if not dense_result.chunks and not lexical_result.chunks:
            return RetrievalResult(chunks=[])

        fused_scores: dict[str, float] = defaultdict(float)
        candidates: dict[str, KnowledgeChunk] = {}

        for result in (dense_result.chunks, lexical_result.chunks):
            for rank, chunk in enumerate(result, start=1):
                fused_scores[chunk.chunk_id] += 1.0 / (self._fusion_k + rank)
                candidates.setdefault(chunk.chunk_id, chunk)

        reranked_chunks = sorted(
            candidates.values(),
            key=lambda chunk: (
                fused_scores[chunk.chunk_id],
                chunk.score or 0.0,
            ),
            reverse=True,
        )

        fused = [
            KnowledgeChunk(
                chunk_id=chunk.chunk_id,
                title=chunk.title,
                source=chunk.source,
                text=chunk.text,
                embedding_text=chunk.embedding_text,
                metadata={
                    **chunk.metadata,
                    "retrieval_strategy": "hybrid_rrf",
                },
                score=fused_scores[chunk.chunk_id],
            )
            for chunk in reranked_chunks[:limit]
        ]
        return RetrievalResult(chunks=fused)
