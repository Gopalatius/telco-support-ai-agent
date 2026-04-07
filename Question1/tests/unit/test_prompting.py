from telco_agent.domain.models import ChatMessage, KnowledgeChunk
from telco_agent.domain.prompting import (
    build_history,
    build_retrieval_context,
    build_retrieval_query_prompt,
    build_user_prompt,
)


def test_build_history_limits_to_recent_turns() -> None:
    history = [
        ChatMessage(role="user", content="first"),
        ChatMessage(role="assistant", content="second"),
        ChatMessage(role="user", content="third"),
    ]

    result = build_history(history, max_turns=2)

    assert result == [
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "third"},
    ]


def test_build_retrieval_context_returns_default_when_empty() -> None:
    assert (
        build_retrieval_context([])
        == "No relevant knowledge base context was retrieved."
    )


def test_build_retrieval_context_serializes_chunks() -> None:
    result = build_retrieval_context(
        [
            KnowledgeChunk(
                chunk_id="billing-1",
                title="Billing Policy",
                source="billing_policy.md",
                text="Late fee is IDR 50,000.",
                embedding_text="embedding",
            )
        ]
    )

    assert "[chunk_id=billing-1]" in result
    assert "source=billing_policy.md" in result
    assert "title=Billing Policy" in result


def test_build_user_prompt_includes_message_and_context() -> None:
    prompt = build_user_prompt(
        "What is the late fee?",
        [
            KnowledgeChunk(
                chunk_id="billing-1",
                title="Billing Policy",
                source="billing_policy.md",
                text="Late fee is IDR 50,000.",
                embedding_text="embedding",
            )
        ],
    )

    assert "Customer message:" in prompt
    assert "What is the late fee?" in prompt
    assert "Knowledge base context:" in prompt
    assert "Use the knowledge base context as the factual basis" in prompt


def test_build_retrieval_query_prompt_handles_empty_and_populated_history() -> None:
    empty_prompt = build_retrieval_query_prompt("latest", [], max_turns=3)
    full_prompt = build_retrieval_query_prompt(
        "latest",
        [
            ChatMessage(role="user", content="first"),
            ChatMessage(role="assistant", content="second"),
        ],
        max_turns=3,
    )

    assert "No prior conversation." in empty_prompt
    assert "Latest customer message:\nlatest" in empty_prompt
    assert "user: first" in full_prompt
    assert "assistant: second" in full_prompt
