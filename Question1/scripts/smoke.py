from qdrant_client import QdrantClient

from telco_agent.application.chat_service import ChatService
from telco_agent.application.ingestion_service import IngestionService
from telco_agent.domain.knowledge_base import load_knowledge_chunks
from telco_agent.domain.models import ChatMessage, ChatRequest
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


def main() -> None:
    settings = Settings()
    embedding_client = GeminiEmbeddingClient(settings)
    generator_client = OpenRouterGeneratorClient(settings)
    vector_store = QdrantKnowledgeStore(
        settings=settings, client=QdrantClient(location=":memory:")
    )

    ingested = IngestionService(
        embedding_client=embedding_client,
        vector_store=vector_store,
    ).ingest(settings.knowledge_base_dir)
    print("Ingested:", ingested)
    print("Query rewrite model:", settings.openrouter_query_rewrite_model)
    print(
        "Query rewrite provider order:",
        settings.openrouter_query_rewrite_provider_order_list,
    )
    print("Answer model:", settings.openrouter_model)
    print("Answer provider order:", settings.openrouter_provider_order_list)

    plan = generator_client.compose_retrieval_query(
        message="Can I still dispute it?",
        history=[
            ChatMessage(
                role="user", content="My invoice from two weeks ago looks wrong."
            ),
            ChatMessage(
                role="assistant", content="I can help with billing policy questions."
            ),
        ],
    )
    print("Retrieval query:", plan.search_query)
    print("Conversation focus:", plan.conversation_focus)

    dense_retriever = DenseRetriever(vector_store)
    if settings.retrieval_strategy == "hybrid":
        retriever = HybridRetriever(
            dense_retriever=dense_retriever,
            lexical_retriever=BM25Retriever(
                load_knowledge_chunks(settings.knowledge_base_dir)
            ),
            candidate_pool=settings.retrieval_candidate_pool,
            fusion_k=settings.retrieval_fusion_k,
        )
    else:
        retriever = dense_retriever

    reranker = OpenRouterReranker(settings) if settings.rerank_enabled else None

    chat_service = ChatService(
        embedding_client=embedding_client,
        generator_client=generator_client,
        retriever=retriever,
        reranker=reranker,
        retrieval_top_k=settings.retrieval_top_k,
        retrieval_min_score=settings.retrieval_min_score,
        rerank_candidate_pool=settings.rerank_candidate_pool,
        rerank_top_n=settings.rerank_top_n,
    )

    supported_response = chat_service.reply(
        ChatRequest(
            message="How much is the late payment fee?",
            history=[],
            session_id="supported-smoke",
        )
    )
    print("Supported reply:", supported_response.reply)
    print("Supported escalate:", supported_response.escalate)

    unsupported_response = chat_service.reply(
        ChatRequest(
            message="Do you support eSIM activation fees?",
            history=[],
            session_id="unsupported-smoke",
        )
    )
    print("Unsupported reply:", unsupported_response.reply)
    print("Unsupported escalate:", unsupported_response.escalate)


if __name__ == "__main__":
    main()
