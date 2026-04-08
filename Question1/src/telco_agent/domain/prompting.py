from textwrap import dedent

from telco_agent.domain.models import ChatMessage, KnowledgeChunk

SYSTEM_PROMPT = dedent(
    """
    You are a telco customer service AI assistant.

    Your job:
    - Help customers with billing, service plans, and basic troubleshooting.
    - Use the provided knowledge base context and conversation history as your
      factual basis.
    - When the evidence fully supports the answer, respond directly and clearly.
    - When the evidence is incomplete, explain that a human agent should take
      over and set escalate to true.
    - Keep responses concise, clear, and helpful.

    Response rules:
    - Return valid JSON only.
    - The reply must be customer-facing.
    - Set escalate to true if a human agent should take over.
    - confidence must be between 0 and 1.
    - rationale is for internal auditing and should be short.
    """
).strip()

RETRIEVAL_QUERY_SYSTEM_PROMPT = dedent(
    """
    You create retrieval queries for a telco customer-service knowledge base.

    Your job:
    - Read the latest customer message and recent conversation history.
    - Resolve references like "that bill", "that plan", or "my issue".
    - Produce a short search query that captures the user's current need.
    - Focus on factual lookup terms, not conversational filler.

    Output rules:
    - Return valid JSON only.
    - search_query should be compact and retrieval-friendly.
    - conversation_focus should explain the resolved intent in one short sentence.
    """
).strip()


def build_history(history: list[ChatMessage], max_turns: int) -> list[dict[str, str]]:
    recent_history = history[-max_turns:]
    return [
        {"role": message.role, "content": message.content} for message in recent_history
    ]


def build_retrieval_context(chunks: list[KnowledgeChunk]) -> str:
    if not chunks:
        return "No relevant knowledge base context was retrieved."

    parts = [
        dedent(
            f"""
            [chunk_id={chunk.chunk_id}]
            source={chunk.source}
            title={chunk.title}
            text={chunk.text}
            """
        ).strip()
        for chunk in chunks
    ]
    return "\n\n".join(parts)


def build_user_prompt(message: str, chunks: list[KnowledgeChunk]) -> str:
    return dedent(
        f"""
        Customer message:
        {message}

        Knowledge base context:
        {build_retrieval_context(chunks)}

        Use the knowledge base context as the factual basis for the reply. When
        the context fully supports the answer, respond directly. When the context
        is incomplete or not relevant enough, return a short escalation-friendly
        reply and set escalate to true.
        """
    ).strip()


def build_retrieval_query_prompt(
    message: str, history: list[ChatMessage], max_turns: int
) -> str:
    history_lines = [f"{item.role}: {item.content}" for item in history[-max_turns:]]

    serialized_history = (
        "\n".join(history_lines) if history_lines else "No prior conversation."
    )
    return dedent(
        f"""
        Recent conversation:
        {serialized_history}

        Latest customer message:
        {message}
        """
    ).strip()
