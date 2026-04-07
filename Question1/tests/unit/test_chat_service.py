from telco_agent.application.chat_service import ChatService
from telco_agent.domain.models import (
    ChatMessage,
    ChatRequest,
    KnowledgeChunk,
    LlmDecision,
    RetrievalQueryPlan,
    RetrievalResult,
)


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.last_query: str | None = None

    def embed_query(self, text: str) -> list[float]:
        self.last_query = text
        return [0.1, 0.2, 0.3]

    def embed_documents(self, chunks: list[KnowledgeChunk]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in chunks]


class FakeGeneratorClient:
    def __init__(
        self,
        *,
        retrieval_query: str = "task: question answering | query: late fee",
    ) -> None:
        self.retrieval_query = retrieval_query

    def compose_retrieval_query(
        self,
        *,
        message: str,
        history: list[ChatMessage],
    ) -> RetrievalQueryPlan:
        return RetrievalQueryPlan(
            search_query=self.retrieval_query,
            conversation_focus=f"{message} ({len(history)})",
        )

    def generate_reply(
        self,
        *,
        message: str,
        history: list[ChatMessage],
        retrieved_chunks: list[KnowledgeChunk],
    ) -> LlmDecision:
        return LlmDecision(
            reply=f"Found {len(retrieved_chunks)} relevant chunks.",
            escalate=False,
            confidence=0.95,
            rationale="relevant",
        )


class EmptyRetriever:
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


class MatchingRetriever(EmptyRetriever):
    def search(
        self,
        *,
        query_text: str,
        query_vector: list[float],
        limit: int,
        min_score: float,
    ) -> RetrievalResult:
        del query_text, query_vector, limit, min_score
        return RetrievalResult(
            chunks=[
                KnowledgeChunk(
                    chunk_id="billing-policy-2",
                    title="Billing Policy",
                    source="billing_policy.md",
                    text="Late payment fee of IDR 50,000 applies after 14 days overdue",
                    embedding_text="",
                    score=0.92,
                )
            ]
        )


class FakeReranker:
    def __init__(self, *, should_fail: bool = False) -> None:
        self.should_fail = should_fail
        self.last_query: str | None = None
        self.last_chunk_ids: list[str] = []
        self.last_top_n: int | None = None

    def rerank(
        self,
        *,
        query_text: str,
        chunks: list[KnowledgeChunk],
        top_n: int,
    ) -> RetrievalResult:
        if self.should_fail:
            raise RuntimeError("boom")
        self.last_query = query_text
        self.last_chunk_ids = [chunk.chunk_id for chunk in chunks]
        self.last_top_n = top_n
        return RetrievalResult(chunks=chunks[:top_n])


def test_chat_service_escalates_when_no_relevant_chunks_are_found() -> None:
    service = ChatService(
        embedding_client=FakeEmbeddingClient(),
        generator_client=FakeGeneratorClient(),
        retriever=EmptyRetriever(),
        reranker=None,
        retrieval_top_k=3,
        retrieval_min_score=0.65,
        rerank_candidate_pool=8,
        rerank_top_n=3,
    )

    response = service.reply(
        ChatRequest(message="Can you explain roaming fees?", history=[])
    )

    assert response.escalate is True
    assert "escalating" in response.reply


def test_chat_service_uses_model_composed_query_for_retrieval() -> None:
    embedding_client = FakeEmbeddingClient()
    service = ChatService(
        embedding_client=embedding_client,
        generator_client=FakeGeneratorClient(
            retrieval_query="task: question answering | query: billing late payment fee"
        ),
        retriever=MatchingRetriever(),
        reranker=None,
        retrieval_top_k=3,
        retrieval_min_score=0.65,
        rerank_candidate_pool=8,
        rerank_top_n=3,
    )

    response = service.reply(
        ChatRequest(
            message="What was the late fee again?",
            history=[ChatMessage(role="user", content="I forgot to pay last month.")],
        )
    )

    assert (
        embedding_client.last_query
        == "task: question answering | query: billing late payment fee"
    )
    assert response.escalate is False
    assert response.reply == "Found 1 relevant chunks."


def test_chat_service_uses_reranker_when_configured() -> None:
    reranker = FakeReranker()
    service = ChatService(
        embedding_client=FakeEmbeddingClient(),
        generator_client=FakeGeneratorClient(
            retrieval_query="task: question answering | query: call 123 billing errors"
        ),
        retriever=MatchingRetriever(),
        reranker=reranker,
        retrieval_top_k=1,
        retrieval_min_score=0.65,
        rerank_candidate_pool=8,
        rerank_top_n=1,
    )

    response = service.reply(
        ChatRequest(
            message="How do I report billing errors?",
            history=[],
        )
    )

    assert (
        reranker.last_query
        == "task: question answering | query: call 123 billing errors"
    )
    assert reranker.last_chunk_ids == ["billing-policy-2"]
    assert reranker.last_top_n == 1
    assert response.escalate is False


def test_chat_service_falls_back_to_retrieval_results_when_reranker_fails() -> None:
    service = ChatService(
        embedding_client=FakeEmbeddingClient(),
        generator_client=FakeGeneratorClient(),
        retriever=MatchingRetriever(),
        reranker=FakeReranker(should_fail=True),
        retrieval_top_k=1,
        retrieval_min_score=0.65,
        rerank_candidate_pool=8,
        rerank_top_n=1,
    )

    response = service.reply(ChatRequest(message="late fee", history=[]))

    assert response.reply == "Found 1 relevant chunks."
    assert response.escalate is False
