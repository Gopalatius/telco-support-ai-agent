from pathlib import Path

from telco_agent.settings import Settings


def make_settings(*, override: str | None = None) -> Settings:
    return Settings.model_construct(
        openrouter_api_key="test",
        gemini_api_key="test",
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_model="moonshotai/kimi-k2.5",
        openrouter_provider_order="moonshotai/int4, moonshotai",
        openrouter_provider_allow_fallbacks=True,
        openrouter_provider_require_parameters=True,
        openrouter_query_rewrite_model="openai/gpt-oss-20b",
        openrouter_query_rewrite_provider_order="groq, together",
        openrouter_query_rewrite_allow_fallbacks=True,
        openrouter_query_rewrite_require_parameters=True,
        embedding_model="gemini-embedding-2-preview",
        qdrant_url="http://localhost:6333",
        qdrant_collection="telco_knowledge_base",
        retrieval_top_k=3,
        retrieval_min_score=0.65,
        retrieval_strategy="dense",
        retrieval_candidate_pool=6,
        retrieval_fusion_k=60,
        rerank_enabled=False,
        rerank_model="cohere/rerank-4-fast",
        rerank_top_n=3,
        rerank_candidate_pool=8,
        max_history_turns=6,
        knowledge_base_dir_override=override,
        app_name="test",
    )


def test_settings_split_provider_orders() -> None:
    settings = make_settings()

    assert settings.openrouter_provider_order_list == ["moonshotai/int4", "moonshotai"]
    assert settings.openrouter_query_rewrite_provider_order_list == ["groq", "together"]


def test_settings_uses_override_for_knowledge_base_dir(tmp_path) -> None:
    override = tmp_path / "kb"
    override.mkdir()

    settings = make_settings(override=str(override))

    assert settings.knowledge_base_dir == override.resolve()


def test_settings_prefers_cwd_knowledge_base_dir(tmp_path, monkeypatch) -> None:
    cwd_data_dir = tmp_path / "data" / "knowledge_base"
    cwd_data_dir.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    settings = make_settings()

    assert settings.knowledge_base_dir == cwd_data_dir


def test_settings_returns_first_candidate_when_no_dir_exists(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    settings = make_settings()
    monkeypatch.setattr(Path, "exists", lambda self: False)

    expected = Path(tmp_path) / "data" / "knowledge_base"
    assert settings.knowledge_base_dir == expected
