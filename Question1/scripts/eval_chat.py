import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from qdrant_client import QdrantClient

from telco_agent.domain.models import ChatMessage, ChatResponse, RetrievalResult
from telco_agent.infrastructure.embeddings import GeminiEmbeddingClient
from telco_agent.infrastructure.openrouter import OpenRouterGeneratorClient
from telco_agent.infrastructure.qdrant_store import QdrantKnowledgeStore
from telco_agent.infrastructure.rerank import OpenRouterReranker
from telco_agent.infrastructure.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
)
from telco_agent.settings import Settings


@dataclass(slots=True)
class ChatEvalCase:
    case_id: str
    message: str
    history: list[ChatMessage]
    expected_escalate: bool
    required_substrings: list[str]


def load_chat_eval_cases(path: Path) -> list[ChatEvalCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        ChatEvalCase(
            case_id=item["id"],
            message=item["message"],
            history=[
                ChatMessage.model_validate(message) for message in item["history"]
            ],
            expected_escalate=item["expected_escalate"],
            required_substrings=item["required_substrings"],
        )
        for item in payload
    ]


def contains_required_substrings(reply: str, required_substrings: list[str]) -> bool:
    normalized_reply = reply.lower()
    return all(fragment.lower() in normalized_reply for fragment in required_substrings)


def run_case(
    *,
    case: ChatEvalCase,
    embedding_client: GeminiEmbeddingClient,
    generator_client: OpenRouterGeneratorClient,
    retriever,
    reranker,
    settings: Settings,
) -> tuple[ChatResponse, dict[str, float], int]:
    timings_ms: dict[str, float] = {}
    total_search_units = 0

    started_at = perf_counter()

    rewrite_started = perf_counter()
    query_plan = generator_client.compose_retrieval_query(
        message=case.message,
        history=case.history,
    )
    timings_ms["query_rewrite_ms"] = round((perf_counter() - rewrite_started) * 1000, 2)

    embedding_started = perf_counter()
    query_vector = embedding_client.embed_query(query_plan.search_query)
    timings_ms["embedding_ms"] = round((perf_counter() - embedding_started) * 1000, 2)

    retrieval_started = perf_counter()
    retrieval_limit = (
        max(settings.retrieval_top_k, settings.rerank_candidate_pool)
        if reranker is not None
        else settings.retrieval_top_k
    )
    retrieval_result = retriever.search(
        query_text=query_plan.search_query,
        query_vector=query_vector,
        limit=retrieval_limit,
        min_score=settings.retrieval_min_score,
    )
    timings_ms["retrieval_ms"] = round((perf_counter() - retrieval_started) * 1000, 2)

    rerank_started = perf_counter()
    if reranker is not None and retrieval_result.has_relevant_chunks:
        retrieval_result = reranker.rerank(
            query_text=query_plan.search_query,
            chunks=retrieval_result.chunks,
            top_n=min(settings.retrieval_top_k, settings.rerank_top_n),
        )
        total_search_units += getattr(reranker, "last_search_units", 0)
    elif retrieval_result.has_relevant_chunks:
        retrieval_result = RetrievalResult(
            chunks=retrieval_result.chunks[: settings.retrieval_top_k]
        )
    timings_ms["rerank_ms"] = round((perf_counter() - rerank_started) * 1000, 2)

    if not retrieval_result.has_relevant_chunks:
        timings_ms["generation_ms"] = 0.0
        timings_ms["total_ms"] = round((perf_counter() - started_at) * 1000, 2)
        return (
            ChatResponse(
                reply=(
                    "I can't confirm that from the current knowledge base, so I'm escalating "
                    "this to a human agent."
                ),
                escalate=True,
            ),
            timings_ms,
            total_search_units,
        )

    generation_started = perf_counter()
    decision = generator_client.generate_reply(
        message=case.message,
        history=case.history,
        retrieved_chunks=retrieval_result.chunks,
    )
    timings_ms["generation_ms"] = round((perf_counter() - generation_started) * 1000, 2)
    timings_ms["total_ms"] = round((perf_counter() - started_at) * 1000, 2)

    return (
        ChatResponse(reply=decision.reply, escalate=decision.escalate),
        timings_ms,
        total_search_units,
    )


def average(values: list[float]) -> float:
    return round(sum(values) / len(values), 2) if values else 0.0


def main() -> None:
    settings = Settings()
    embedding_client = GeminiEmbeddingClient(settings)
    generator_client = OpenRouterGeneratorClient(settings)
    vector_store = QdrantKnowledgeStore(
        settings=settings, client=QdrantClient(location=":memory:")
    )

    from telco_agent.domain.knowledge_base import load_knowledge_chunks

    chunks = load_knowledge_chunks(settings.knowledge_base_dir)
    vector_store.recreate_collection()
    vector_store.upsert_chunks(chunks, embedding_client.embed_documents(chunks))

    dense = DenseRetriever(vector_store)
    retriever = (
        HybridRetriever(
            dense_retriever=dense,
            lexical_retriever=BM25Retriever(chunks),
            candidate_pool=settings.retrieval_candidate_pool,
            fusion_k=settings.retrieval_fusion_k,
        )
        if settings.retrieval_strategy == "hybrid"
        else dense
    )
    reranker = (
        OpenRouterReranker(settings=settings) if settings.rerank_enabled else None
    )

    cases = load_chat_eval_cases(
        Path(__file__).resolve().parents[1] / "data" / "evals" / "chat_eval.json"
    )

    pass_count = 0
    failures: list[dict[str, object]] = []
    totals: dict[str, list[float]] = {
        "query_rewrite_ms": [],
        "embedding_ms": [],
        "retrieval_ms": [],
        "rerank_ms": [],
        "generation_ms": [],
        "total_ms": [],
        "supported_total_ms": [],
        "unsupported_total_ms": [],
    }
    total_search_units = 0

    for case in cases:
        response, timings_ms, search_units = run_case(
            case=case,
            embedding_client=embedding_client,
            generator_client=generator_client,
            retriever=retriever,
            reranker=reranker,
            settings=settings,
        )
        total_search_units += search_units
        for key, value in timings_ms.items():
            totals[key].append(value)
        if case.expected_escalate:
            totals["unsupported_total_ms"].append(timings_ms["total_ms"])
        else:
            totals["supported_total_ms"].append(timings_ms["total_ms"])

        passed = (
            response.escalate == case.expected_escalate
            and contains_required_substrings(response.reply, case.required_substrings)
        )
        pass_count += int(passed)

        if not passed:
            failures.append(
                {
                    "case_id": case.case_id,
                    "message": case.message,
                    "expected_escalate": case.expected_escalate,
                    "actual_escalate": response.escalate,
                    "required_substrings": case.required_substrings,
                    "reply": response.reply,
                    "timings_ms": timings_ms,
                }
            )

    payload = {
        "cases": len(cases),
        "pass_rate": round(pass_count / len(cases), 4),
        "avg_query_rewrite_ms": average(totals["query_rewrite_ms"]),
        "avg_embedding_ms": average(totals["embedding_ms"]),
        "avg_retrieval_ms": average(totals["retrieval_ms"]),
        "avg_rerank_ms": average(totals["rerank_ms"]),
        "avg_generation_ms": average(totals["generation_ms"]),
        "avg_total_ms": average(totals["total_ms"]),
        "avg_supported_total_ms": average(totals["supported_total_ms"]),
        "avg_unsupported_total_ms": average(totals["unsupported_total_ms"]),
        "search_units": total_search_units,
        "failures": failures,
    }

    output_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "evals"
        / "chat_benchmark_latest.json"
    )
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
