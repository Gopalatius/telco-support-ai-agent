from telco_agent.domain.models import (
    ChatRequest,
    ChatResponse,
    EmbeddingClient,
    GeneratorClient,
    Reranker,
    RetrievalResult,
    Retriever,
)


class ChatService:
    def __init__(
        self,
        *,
        embedding_client: EmbeddingClient,
        generator_client: GeneratorClient,
        retriever: Retriever,
        reranker: Reranker | None,
        retrieval_top_k: int,
        retrieval_min_score: float,
        rerank_candidate_pool: int,
        rerank_top_n: int,
    ) -> None:
        self._embedding_client = embedding_client
        self._generator_client = generator_client
        self._retriever = retriever
        self._reranker = reranker
        self._retrieval_top_k = retrieval_top_k
        self._retrieval_min_score = retrieval_min_score
        self._rerank_candidate_pool = rerank_candidate_pool
        self._rerank_top_n = rerank_top_n

    def reply(self, request: ChatRequest) -> ChatResponse:
        query_plan = self._generator_client.compose_retrieval_query(
            message=request.message,
            history=request.history,
        )
        query_vector = self._embedding_client.embed_query(query_plan.search_query)
        retrieval_limit = (
            max(self._retrieval_top_k, self._rerank_candidate_pool)
            if self._reranker is not None
            else self._retrieval_top_k
        )
        retrieval_result = self._retriever.search(
            query_text=query_plan.search_query,
            query_vector=query_vector,
            limit=retrieval_limit,
            min_score=self._retrieval_min_score,
        )

        if self._reranker is not None and retrieval_result.has_relevant_chunks:
            try:
                retrieval_result = self._reranker.rerank(
                    query_text=query_plan.search_query,
                    chunks=retrieval_result.chunks,
                    top_n=min(self._retrieval_top_k, self._rerank_top_n),
                )
            except RuntimeError:
                retrieval_result = RetrievalResult(
                    chunks=retrieval_result.chunks[: self._retrieval_top_k]
                )
        elif retrieval_result.has_relevant_chunks:
            retrieval_result = RetrievalResult(
                chunks=retrieval_result.chunks[: self._retrieval_top_k]
            )

        if not retrieval_result.has_relevant_chunks:
            return ChatResponse(
                reply=(
                    "I can't confirm that from the current knowledge base, so I'm escalating "
                    "this to a human agent."
                ),
                escalate=True,
            )

        decision = self._generator_client.generate_reply(
            message=request.message,
            history=request.history,
            retrieved_chunks=retrieval_result.chunks,
        )

        return ChatResponse(reply=decision.reply, escalate=decision.escalate)
