import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from qdrant_client import QdrantClient

from telco_agent.domain.knowledge_base import (
    format_query_for_embedding,
    load_knowledge_chunks,
)
from telco_agent.domain.models import RetrievalResult
from telco_agent.infrastructure.embeddings import GeminiEmbeddingClient
from telco_agent.infrastructure.qdrant_store import QdrantKnowledgeStore
from telco_agent.infrastructure.rerank import OpenRouterReranker
from telco_agent.infrastructure.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
)
from telco_agent.settings import Settings


@dataclass(slots=True)
class EvalCase:
    case_id: str
    query: str
    expected_chunk_ids: list[str]


def load_eval_cases(path: Path) -> list[EvalCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        EvalCase(
            case_id=item["id"],
            query=item["query"],
            expected_chunk_ids=item["expected_chunk_ids"],
        )
        for item in payload
    ]


def evaluate_retriever(
    *,
    name: str,
    retriever,
    embedding_client: GeminiEmbeddingClient,
    cases: list[EvalCase],
    reranker=None,
    rerank_top_n: int = 3,
    rerank_candidate_pool: int = 8,
) -> dict[str, object]:
    top1_hits = 0
    top3_hits = 0
    failures: list[dict[str, object]] = []
    latency_ms: list[float] = []
    total_search_units = 0

    for case in cases:
        started_at = perf_counter()
        query_vector = embedding_client.embed_query(
            format_query_for_embedding(case.query)
        )
        result = retriever.search(
            query_text=case.query,
            query_vector=query_vector,
            limit=max(3, rerank_candidate_pool) if reranker is not None else 3,
            min_score=0.65,
        )
        if reranker is not None and result.has_relevant_chunks:
            result = reranker.rerank(
                query_text=case.query,
                chunks=result.chunks,
                top_n=min(3, rerank_top_n),
            )
            total_search_units += getattr(reranker, "last_search_units", 0)
        elif result.has_relevant_chunks:
            result = RetrievalResult(chunks=result.chunks[:3])

        latency_ms.append(round((perf_counter() - started_at) * 1000, 2))
        ranked_chunk_ids = [chunk.chunk_id for chunk in result.chunks]

        top1 = bool(
            ranked_chunk_ids[:1] and ranked_chunk_ids[0] in case.expected_chunk_ids
        )
        top3 = any(
            chunk_id in case.expected_chunk_ids for chunk_id in ranked_chunk_ids[:3]
        )

        top1_hits += int(top1)
        top3_hits += int(top3)

        if not top1 or not top3:
            failures.append(
                {
                    "case_id": case.case_id,
                    "query": case.query,
                    "expected": case.expected_chunk_ids,
                    "retrieved": ranked_chunk_ids,
                }
            )

    total = len(cases)
    return {
        "name": name,
        "top1_accuracy": round(top1_hits / total, 4),
        "top3_recall": round(top3_hits / total, 4),
        "avg_latency_ms": round(sum(latency_ms) / total, 2),
        "search_units": total_search_units,
        "failures": failures,
    }


def main() -> None:
    settings = Settings()
    chunks = load_knowledge_chunks(settings.knowledge_base_dir)
    embedding_client = GeminiEmbeddingClient(settings)
    vector_store = QdrantKnowledgeStore(
        settings=settings, client=QdrantClient(location=":memory:")
    )
    vector_store.recreate_collection()
    vector_store.upsert_chunks(chunks, embedding_client.embed_documents(chunks))

    dense = DenseRetriever(vector_store)
    hybrid = HybridRetriever(
        dense_retriever=dense,
        lexical_retriever=BM25Retriever(chunks),
        candidate_pool=settings.retrieval_candidate_pool,
        fusion_k=settings.retrieval_fusion_k,
    )
    reranker = OpenRouterReranker(settings=settings)

    cases = load_eval_cases(
        Path(__file__).resolve().parents[1] / "data" / "evals" / "retrieval_eval.json"
    )
    results = [
        evaluate_retriever(
            name="dense",
            retriever=dense,
            embedding_client=embedding_client,
            cases=cases,
        ),
        evaluate_retriever(
            name="hybrid_rrf",
            retriever=hybrid,
            embedding_client=embedding_client,
            cases=cases,
        ),
        evaluate_retriever(
            name="dense_rerank",
            retriever=dense,
            embedding_client=embedding_client,
            cases=cases,
            reranker=reranker,
            rerank_top_n=settings.rerank_top_n,
            rerank_candidate_pool=settings.rerank_candidate_pool,
        ),
        evaluate_retriever(
            name="hybrid_rrf_rerank",
            retriever=hybrid,
            embedding_client=embedding_client,
            cases=cases,
            reranker=reranker,
            rerank_top_n=settings.rerank_top_n,
            rerank_candidate_pool=settings.rerank_candidate_pool,
        ),
    ]

    payload = {"cases": len(cases), "results": results}
    output_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "evals"
        / "retrieval_benchmark_latest.json"
    )
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
