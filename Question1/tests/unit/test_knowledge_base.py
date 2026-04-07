from pathlib import Path

from telco_agent.domain.knowledge_base import (
    format_document_for_embedding,
    format_query_for_embedding,
    load_knowledge_chunks,
    slugify,
)


def test_load_knowledge_chunks_splits_bullets_into_atomic_chunks() -> None:
    knowledge_dir = Path(__file__).resolve().parents[2] / "data" / "knowledge_base"

    chunks = load_knowledge_chunks(knowledge_dir)

    assert len(chunks) == 12
    assert chunks[0].chunk_id == "billing-policy-1"
    assert chunks[0].title == "Billing Policy"


def test_load_knowledge_chunks_uses_body_when_file_has_no_bullets(tmp_path) -> None:
    knowledge_dir = tmp_path / "kb"
    knowledge_dir.mkdir()
    (knowledge_dir / "faq.md").write_text(
        "# FAQ\nThis is body text only.\n", encoding="utf-8"
    )

    chunks = load_knowledge_chunks(knowledge_dir)

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "faq-1"
    assert chunks[0].text == "This is body text only."


def test_embedding_formats_match_gemini_embeddings_two_guidance() -> None:
    assert (
        format_query_for_embedding("late fee")
        == "task: question answering | query: late fee"
    )
    assert (
        format_document_for_embedding("Billing Policy", "Late payment fee applies")
        == "title: Billing Policy | text: Late payment fee applies"
    )


def test_slugify_returns_chunk_for_empty_slug() -> None:
    assert slugify("!!!") == "chunk"
