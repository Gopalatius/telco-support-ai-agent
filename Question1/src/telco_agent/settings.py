from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openrouter_api_key: str = Field(alias="OPENROUTER_API_KEY")
    gemini_api_key: str = Field(alias="GEMINI_API_KEY")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        alias="OPENROUTER_BASE_URL",
    )
    openrouter_model: str = Field(
        default="moonshotai/kimi-k2.5", alias="OPENROUTER_MODEL"
    )
    openrouter_provider_order: str | None = Field(
        default="moonshotai/int4",
        alias="OPENROUTER_PROVIDER_ORDER",
    )
    openrouter_provider_allow_fallbacks: bool = Field(
        default=True,
        alias="OPENROUTER_PROVIDER_ALLOW_FALLBACKS",
    )
    openrouter_provider_require_parameters: bool = Field(
        default=True,
        alias="OPENROUTER_PROVIDER_REQUIRE_PARAMETERS",
    )
    openrouter_query_rewrite_model: str = Field(
        default="openai/gpt-oss-20b",
        alias="OPENROUTER_QUERY_REWRITE_MODEL",
    )
    openrouter_query_rewrite_provider_order: str | None = Field(
        default="groq",
        alias="OPENROUTER_QUERY_REWRITE_PROVIDER_ORDER",
    )
    openrouter_query_rewrite_allow_fallbacks: bool = Field(
        default=True,
        alias="OPENROUTER_QUERY_REWRITE_ALLOW_FALLBACKS",
    )
    openrouter_query_rewrite_require_parameters: bool = Field(
        default=True,
        alias="OPENROUTER_QUERY_REWRITE_REQUIRE_PARAMETERS",
    )
    embedding_model: str = Field(
        default="gemini-embedding-2-preview", alias="EMBEDDING_MODEL"
    )
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_collection: str = Field(
        default="telco_knowledge_base", alias="QDRANT_COLLECTION"
    )
    retrieval_top_k: int = Field(default=3, alias="RETRIEVAL_TOP_K")
    retrieval_min_score: float = Field(default=0.65, alias="RETRIEVAL_MIN_SCORE")
    retrieval_strategy: str = Field(default="dense", alias="RETRIEVAL_STRATEGY")
    retrieval_candidate_pool: int = Field(default=6, alias="RETRIEVAL_CANDIDATE_POOL")
    retrieval_fusion_k: int = Field(default=60, alias="RETRIEVAL_FUSION_K")
    rerank_enabled: bool = Field(default=False, alias="RERANK_ENABLED")
    rerank_model: str = Field(default="cohere/rerank-4-fast", alias="RERANK_MODEL")
    rerank_top_n: int = Field(default=3, alias="RERANK_TOP_N")
    rerank_candidate_pool: int = Field(default=8, alias="RERANK_CANDIDATE_POOL")
    max_history_turns: int = Field(default=6, alias="MAX_HISTORY_TURNS")
    knowledge_base_dir_override: str | None = Field(
        default=None, alias="KNOWLEDGE_BASE_DIR"
    )
    app_name: str = "telco-customer-service-agent"

    model_config = SettingsConfigDict(
        env_file=(
            ".env",
            "../.env",
        ),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @staticmethod
    def _split_provider_order(value: str | None) -> list[str] | None:
        if value is None:
            return None

        items = [item.strip() for item in value.split(",") if item.strip()]
        return items or None

    @property
    def openrouter_provider_order_list(self) -> list[str] | None:
        return self._split_provider_order(self.openrouter_provider_order)

    @property
    def openrouter_query_rewrite_provider_order_list(self) -> list[str] | None:
        return self._split_provider_order(self.openrouter_query_rewrite_provider_order)

    @property
    def knowledge_base_dir(self) -> Path:
        if self.knowledge_base_dir_override:
            return Path(self.knowledge_base_dir_override).resolve()

        candidates = [
            Path.cwd() / "data" / "knowledge_base",
            Path(__file__).resolve().parents[2] / "data" / "knowledge_base",
            Path("/app/data/knowledge_base"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        return candidates[0]
