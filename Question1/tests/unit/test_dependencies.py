from pathlib import Path

from telco_agent.api import dependencies
from telco_agent.settings import Settings


class StubEmbeddingClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings


class StubGeneratorClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings


class StubVectorStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings


class StubDenseRetriever:
    def __init__(self, vector_store) -> None:
        self.vector_store = vector_store


class StubBM25Retriever:
    def __init__(self, chunks) -> None:
        self.chunks = chunks


class StubHybridRetriever:
    def __init__(
        self,
        *,
        dense_retriever,
        lexical_retriever,
        candidate_pool: int,
        fusion_k: int,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.lexical_retriever = lexical_retriever
        self.candidate_pool = candidate_pool
        self.fusion_k = fusion_k


class StubReranker:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings


class StubChatService:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class StubIngestionService:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


def clear_dependency_caches() -> None:
    dependencies.get_settings.cache_clear()
    dependencies.get_embedding_client.cache_clear()
    dependencies.get_vector_store.cache_clear()
    dependencies.get_knowledge_chunks.cache_clear()
    dependencies.get_retriever.cache_clear()
    dependencies.get_generator_client.cache_clear()
    dependencies.get_reranker.cache_clear()
    dependencies.get_chat_service.cache_clear()
    dependencies.get_ingestion_service.cache_clear()


def make_settings(
    *, retrieval_strategy: str = "dense", rerank_enabled: bool = False
) -> Settings:
    return Settings.model_construct(
        openrouter_api_key="test",
        gemini_api_key="test",
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_model="moonshotai/kimi-k2.5",
        openrouter_provider_order="moonshotai/int4",
        openrouter_provider_allow_fallbacks=True,
        openrouter_provider_require_parameters=True,
        openrouter_query_rewrite_model="openai/gpt-oss-20b",
        openrouter_query_rewrite_provider_order="groq",
        openrouter_query_rewrite_allow_fallbacks=True,
        openrouter_query_rewrite_require_parameters=True,
        embedding_model="gemini-embedding-2-preview",
        qdrant_url="http://localhost:6333",
        qdrant_collection="telco_knowledge_base",
        retrieval_top_k=3,
        retrieval_min_score=0.65,
        retrieval_strategy=retrieval_strategy,
        retrieval_candidate_pool=6,
        retrieval_fusion_k=60,
        rerank_enabled=rerank_enabled,
        rerank_model="cohere/rerank-4-fast",
        rerank_top_n=3,
        rerank_candidate_pool=8,
        max_history_turns=6,
        knowledge_base_dir_override=str(Path("/tmp/knowledge_base")),
        app_name="test-app",
    )


def test_dependencies_build_dense_stack(monkeypatch) -> None:
    clear_dependency_caches()
    settings = make_settings(retrieval_strategy="dense", rerank_enabled=False)

    monkeypatch.setattr(dependencies, "Settings", lambda: settings)
    monkeypatch.setattr(dependencies, "GeminiEmbeddingClient", StubEmbeddingClient)
    monkeypatch.setattr(dependencies, "QdrantKnowledgeStore", StubVectorStore)
    monkeypatch.setattr(dependencies, "DenseRetriever", StubDenseRetriever)
    monkeypatch.setattr(dependencies, "OpenRouterGeneratorClient", StubGeneratorClient)
    monkeypatch.setattr(dependencies, "ChatService", StubChatService)
    monkeypatch.setattr(dependencies, "IngestionService", StubIngestionService)

    embedding_client = dependencies.get_embedding_client()
    vector_store = dependencies.get_vector_store()
    retriever = dependencies.get_retriever()
    generator_client = dependencies.get_generator_client()
    reranker = dependencies.get_reranker()
    chat_service = dependencies.get_chat_service()
    ingestion_service = dependencies.get_ingestion_service()

    assert isinstance(embedding_client, StubEmbeddingClient)
    assert isinstance(vector_store, StubVectorStore)
    assert isinstance(retriever, StubDenseRetriever)
    assert isinstance(generator_client, StubGeneratorClient)
    assert reranker is None
    assert chat_service.kwargs["retriever"] is retriever
    assert ingestion_service.kwargs["vector_store"] is vector_store

    clear_dependency_caches()


def test_dependencies_build_hybrid_stack_with_reranker(monkeypatch) -> None:
    clear_dependency_caches()
    settings = make_settings(retrieval_strategy="hybrid", rerank_enabled=True)
    chunks = ["chunk-a"]

    monkeypatch.setattr(dependencies, "Settings", lambda: settings)
    monkeypatch.setattr(dependencies, "GeminiEmbeddingClient", StubEmbeddingClient)
    monkeypatch.setattr(dependencies, "QdrantKnowledgeStore", StubVectorStore)
    monkeypatch.setattr(dependencies, "DenseRetriever", StubDenseRetriever)
    monkeypatch.setattr(dependencies, "BM25Retriever", StubBM25Retriever)
    monkeypatch.setattr(dependencies, "HybridRetriever", StubHybridRetriever)
    monkeypatch.setattr(dependencies, "OpenRouterGeneratorClient", StubGeneratorClient)
    monkeypatch.setattr(dependencies, "OpenRouterReranker", StubReranker)
    monkeypatch.setattr(dependencies, "load_knowledge_chunks", lambda _: chunks)
    monkeypatch.setattr(dependencies, "ChatService", StubChatService)

    retriever = dependencies.get_retriever()
    reranker = dependencies.get_reranker()
    chat_service = dependencies.get_chat_service()

    assert isinstance(retriever, StubHybridRetriever)
    assert retriever.lexical_retriever.chunks == chunks
    assert isinstance(reranker, StubReranker)
    assert chat_service.kwargs["reranker"] is reranker

    clear_dependency_caches()
