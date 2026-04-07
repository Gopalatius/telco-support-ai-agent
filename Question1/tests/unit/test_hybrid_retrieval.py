from typing import Any, cast

from telco_agent.domain.models import KnowledgeChunk, RetrievalResult
from telco_agent.infrastructure.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    normalize_for_lexical_search,
)


class StubVectorStore:
    def __init__(self, chunks: list[KnowledgeChunk]) -> None:
        self._chunks = chunks

    def search(
        self, vector: list[float], limit: int, min_score: float
    ) -> RetrievalResult:
        del vector, min_score
        return RetrievalResult(chunks=self._chunks[:limit])

    def recreate_collection(self) -> None:
        return None

    def upsert_chunks(
        self, chunks: list[KnowledgeChunk], vectors: list[list[float]]
    ) -> None:
        del chunks, vectors
        return

    def healthcheck(self) -> bool:
        return True


class EmptyLexicalRetriever:
    def search(
        self,
        *,
        query_text: str,
        query_vector: list[float],
        limit: int,
        min_score: float,
    ) -> RetrievalResult:
        del query_text, query_vector, limit, min_score
        return RetrievalResult(chunks=[])


def test_normalize_for_lexical_search_keeps_alphanumeric_tokens() -> None:
    assert normalize_for_lexical_search("IDR 50,000 late-fee!") == [
        "idr",
        "50",
        "000",
        "late",
        "fee",
    ]


def test_bm25_retriever_returns_empty_result_for_blank_query() -> None:
    result = BM25Retriever(
        [KnowledgeChunk("a", "Billing", "billing.md", "Late fee applies", "")]
    ).search(query_text="!!!", query_vector=[], limit=1, min_score=0.0)

    assert result.chunks == []


def test_bm25_retriever_prefers_exact_term_overlap() -> None:
    chunks = [
        KnowledgeChunk(
            chunk_id="billing-policy-2",
            title="Billing Policy",
            source="billing_policy.md",
            text="Late payment fee of IDR 50,000 applies after 14 days overdue",
            embedding_text="",
        ),
        KnowledgeChunk(
            chunk_id="service-plans-4",
            title="Service Plans",
            source="service_plans.md",
            text="All plans include free access to streaming partners on weekends",
            embedding_text="",
        ),
    ]

    result = BM25Retriever(chunks).search(
        query_text="IDR 50,000 late fee",
        query_vector=[],
        limit=1,
        min_score=0.0,
    )

    assert result.chunks[0].chunk_id == "billing-policy-2"


def test_rrf_promotes_chunks_present_in_both_rankings() -> None:
    dense_chunks = [
        KnowledgeChunk(
            chunk_id="service-plans-2",
            title="Service Plans",
            source="service_plans.md",
            text="Pro Plan: IDR 199,000/month - 50GB data, unlimited calls, 5GB hotspot",
            embedding_text="",
            score=0.80,
        ),
        KnowledgeChunk(
            chunk_id="service-plans-3",
            title="Service Plans",
            source="service_plans.md",
            text="Unlimited Plan: IDR 299,000/month - Unlimited data, calls, and 20GB hotspot",
            embedding_text="",
            score=0.79,
        ),
    ]
    lexical_chunks = [
        KnowledgeChunk(
            chunk_id="service-plans-2",
            title="Service Plans",
            source="service_plans.md",
            text="Pro Plan: IDR 199,000/month - 50GB data, unlimited calls, 5GB hotspot",
            embedding_text="",
        ),
        KnowledgeChunk(
            chunk_id="billing-policy-1",
            title="Billing Policy",
            source="billing_policy.md",
            text="Bills are generated on the 1st of every month",
            embedding_text="",
        ),
    ]

    result = HybridRetriever(
        dense_retriever=DenseRetriever(cast("Any", StubVectorStore(dense_chunks))),
        lexical_retriever=BM25Retriever(lexical_chunks),
        candidate_pool=3,
        fusion_k=60,
    ).search(
        query_text="pro plan 5gb hotspot",
        query_vector=[0.1],
        limit=2,
        min_score=0.0,
    )

    assert result.chunks[0].chunk_id == "service-plans-2"
    assert result.chunks[0].metadata["retrieval_strategy"] == "hybrid_rrf"


def test_rrf_returns_empty_when_both_retrievers_are_empty() -> None:
    result = HybridRetriever(
        dense_retriever=DenseRetriever(cast("Any", StubVectorStore([]))),
        lexical_retriever=EmptyLexicalRetriever(),
        candidate_pool=3,
        fusion_k=60,
    ).search(
        query_text="missing term",
        query_vector=[0.1],
        limit=2,
        min_score=0.0,
    )

    assert result.chunks == []
