from functools import lru_cache
from typing import Any, cast

from telco_agent.application.chat_service import ChatService
from telco_agent.application.ingestion_service import IngestionService
from telco_agent.domain.knowledge_base import load_knowledge_chunks
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


@lru_cache
def get_settings() -> Settings:
    settings_factory = cast("Any", Settings)
    return settings_factory()


@lru_cache
def get_embedding_client() -> GeminiEmbeddingClient:
    return GeminiEmbeddingClient(settings=get_settings())


@lru_cache
def get_vector_store() -> QdrantKnowledgeStore:
    return QdrantKnowledgeStore(settings=get_settings())


@lru_cache
def get_knowledge_chunks():
    return load_knowledge_chunks(get_settings().knowledge_base_dir)


@lru_cache
def get_retriever():
    settings = get_settings()
    dense = DenseRetriever(get_vector_store())

    if settings.retrieval_strategy == "dense":
        return dense

    lexical = BM25Retriever(get_knowledge_chunks())
    return HybridRetriever(
        dense_retriever=dense,
        lexical_retriever=lexical,
        candidate_pool=settings.retrieval_candidate_pool,
        fusion_k=settings.retrieval_fusion_k,
    )


@lru_cache
def get_generator_client() -> OpenRouterGeneratorClient:
    return OpenRouterGeneratorClient(settings=get_settings())


@lru_cache
def get_reranker():
    settings = get_settings()
    if not settings.rerank_enabled:
        return None
    return OpenRouterReranker(settings=settings)


@lru_cache
def get_chat_service() -> ChatService:
    settings = get_settings()
    return ChatService(
        embedding_client=get_embedding_client(),
        generator_client=get_generator_client(),
        retriever=get_retriever(),
        reranker=get_reranker(),
        retrieval_top_k=settings.retrieval_top_k,
        retrieval_min_score=settings.retrieval_min_score,
        rerank_candidate_pool=settings.rerank_candidate_pool,
        rerank_top_n=settings.rerank_top_n,
    )


@lru_cache
def get_ingestion_service() -> IngestionService:
    return IngestionService(
        embedding_client=get_embedding_client(),
        vector_store=get_vector_store(),
    )
