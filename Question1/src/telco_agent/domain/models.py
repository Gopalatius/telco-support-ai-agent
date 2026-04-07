from dataclasses import dataclass, field
from typing import Literal, Protocol

from pydantic import BaseModel, Field

Role = Literal["user", "assistant"]


class ChatMessage(BaseModel):
    role: Role
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)
    session_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    escalate: bool


class LlmDecision(BaseModel):
    reply: str
    escalate: bool
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(default="")


class RetrievalQueryPlan(BaseModel):
    search_query: str = Field(min_length=1)
    conversation_focus: str = Field(min_length=1)


@dataclass(slots=True)
class KnowledgeChunk:
    chunk_id: str
    title: str
    source: str
    text: str
    embedding_text: str
    metadata: dict[str, str] = field(default_factory=dict)
    score: float | None = None


@dataclass(slots=True)
class RetrievalResult:
    chunks: list[KnowledgeChunk]

    @property
    def has_relevant_chunks(self) -> bool:
        return bool(self.chunks)


class EmbeddingClient(Protocol):
    def embed_query(self, text: str) -> list[float]: ...

    def embed_documents(self, chunks: list[KnowledgeChunk]) -> list[list[float]]: ...


class VectorStore(Protocol):
    def recreate_collection(self) -> None: ...

    def upsert_chunks(
        self, chunks: list[KnowledgeChunk], vectors: list[list[float]]
    ) -> None: ...

    def search(
        self, vector: list[float], limit: int, min_score: float
    ) -> RetrievalResult: ...

    def healthcheck(self) -> bool: ...


class Retriever(Protocol):
    def search(
        self,
        *,
        query_text: str,
        query_vector: list[float],
        limit: int,
        min_score: float,
    ) -> RetrievalResult: ...


class Reranker(Protocol):
    def rerank(
        self,
        *,
        query_text: str,
        chunks: list[KnowledgeChunk],
        top_n: int,
    ) -> RetrievalResult: ...


class GeneratorClient(Protocol):
    def compose_retrieval_query(
        self,
        *,
        message: str,
        history: list[ChatMessage],
    ) -> RetrievalQueryPlan: ...

    def generate_reply(
        self,
        *,
        message: str,
        history: list[ChatMessage],
        retrieved_chunks: list[KnowledgeChunk],
    ) -> LlmDecision: ...
