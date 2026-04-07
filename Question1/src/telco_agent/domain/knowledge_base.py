from pathlib import Path
from re import sub

from telco_agent.domain.models import KnowledgeChunk


def slugify(value: str) -> str:
    normalized = sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized or "chunk"


def format_query_for_embedding(message: str) -> str:
    return f"task: question answering | query: {message.strip()}"


def format_document_for_embedding(title: str, text: str) -> str:
    return f"title: {title.strip() or 'none'} | text: {text.strip()}"


def load_knowledge_chunks(knowledge_base_dir: Path) -> list[KnowledgeChunk]:
    chunks: list[KnowledgeChunk] = []

    for path in sorted(knowledge_base_dir.glob("*.md")):
        raw_lines = path.read_text(encoding="utf-8").splitlines()
        lines = [line.strip() for line in raw_lines if line.strip()]
        title = (
            lines[0].removeprefix("# ").strip()
            if lines
            else path.stem.replace("_", " ").title()
        )
        bullet_lines = [
            line.removeprefix("- ").removeprefix("* ").strip()
            for line in lines[1:]
            if line.startswith(("-", "*"))
        ]

        if not bullet_lines:
            body = " ".join(lines[1:])
            bullet_lines = [body] if body else []

        for index, bullet in enumerate(bullet_lines, start=1):
            chunk_id = f"{slugify(path.stem)}-{index}"
            chunks.append(
                KnowledgeChunk(
                    chunk_id=chunk_id,
                    title=title,
                    source=path.name,
                    text=bullet,
                    embedding_text=format_document_for_embedding(
                        title=title, text=bullet
                    ),
                    metadata={
                        "source": path.name,
                        "title": title,
                        "chunk_index": str(index),
                    },
                )
            )

    return chunks
