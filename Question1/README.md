# Question 1: Customer Service AI Agent

This service implements a customer-service AI assistant for a telecommunications company using `FastAPI`, `Granian`, `Qdrant`, `OpenRouter`, and Gemini embeddings.

## Tech Choices

- API server: `FastAPI` with `Granian`
- Retrieval-query rewrite model: `openai/gpt-oss-20b` through the OpenRouter Responses API, with provider preference `groq`
- Generator model: `moonshotai/kimi-k2.5` through the Python `openai` SDK configured with `base_url=https://openrouter.ai/api/v1`, with provider preference `moonshotai/int4`
- Embeddings: `gemini-embedding-2-preview` called directly via the Google GenAI SDK
- Vector store: `Qdrant`
- Retrieval: dense retrieval by default, with optional `BM25 + RRF` hybrid retrieval support
- Optional reranking: `cohere/rerank-4-fast` through OpenRouter
- Tooling: `uv`, `ruff`, `ty`, `pytest`

## Why Kimi

I chose `Kimi K2.5` as the default answer model because it offers a strong quality-to-cost tradeoff, performs well on structured-output style tasks, and benefits from OpenRouter prompt caching support. For the cheaper query-rewrite step, I use `openai/gpt-oss-20b` so the system does not spend premium answer-model budget on a small retrieval-helper task.

At the same time, the application flow is intentionally isolated from provider-specific logic. The routing, orchestration, retrieval, and prompt policy do not depend on OpenRouter or Kimi directly; only the generator adapter does. That keeps the swap surface small if a stricter `Gemini`, `OpenAI`, or `Claude` implementation is preferred later.

## Why This Shape

- Retrieval is app-controlled for determinism, testing, and reliable escalation behavior.
- A model-driven retrieval-query composition step handles long conversations by rewriting the latest turn plus recent history into a focused search query.
- The generator only answers from retrieved chunks and recent history, and the API escalates when retrieval cannot support a confident answer.
- Hybrid retrieval and reranking are implemented behind config, but dense retrieval remains the default because the current labeled eval set shows no measurable lift on this very small corpus.

## Why This Architecture

I used a lightweight clean separation between API, application, domain, and infrastructure concerns for three practical reasons:

- keep FastAPI handlers thin and focused on HTTP concerns
- isolate provider-specific code for the LLM, embeddings, and vector store
- make the orchestration and retrieval logic easy to unit test without network calls

This is not meant as heavyweight enterprise architecture. It is a small, testability-first structure that keeps the main chat flow readable and the swap boundaries explicit.

## Setup

1. Ensure the root `.env` contains valid `OPENROUTER_API_KEY` and `GEMINI_API_KEY`.
2. Install dependencies:

```bash
make install
```

3. Ingest the knowledge base:

```bash
make ingest
```

4. Run the API:

```bash
make run
```

## Docker Workflow

For a reviewer-friendly Docker flow, these are the only commands needed:

```bash
make docker-build
make docker-up
make docker-ingest
make docker-smoke
```

If your environment uses the Compose plugin instead of `docker-compose`, override once:

```bash
make COMPOSE="docker compose" docker-build docker-up docker-ingest docker-smoke
```

What each target does:

- `make docker-build`: builds the app image
- `make docker-up`: starts Qdrant and the API container
- `make docker-ingest`: waits for Qdrant, then ingests the knowledge base through a one-shot container
- `make docker-smoke`: checks `/health` and sends a real `/chat` request
- `make docker-down`: stops the Docker stack

The app and ingest services both wait for Qdrant readiness before starting, so the stack behaves reliably on a cold start.

## Checks

```bash
make test
make coverage
make lint
make typecheck
make smoke
make eval-retrieval
make eval-chat
```

`scripts/smoke.py` uses an in-memory Qdrant instance, so it verifies the real Gemini and OpenRouter integrations without requiring Docker.

`scripts/eval_retrieval.py` runs a 28-case labeled retrieval benchmark stored in `data/evals/retrieval_eval.json` and compares `dense`, `hybrid_rrf`, `dense_rerank`, and `hybrid_rrf_rerank`. The latest measured result is written to `data/evals/retrieval_benchmark_latest.json` so a reviewer can inspect it directly.

`scripts/eval_chat.py` runs a chat-quality benchmark stored in `data/evals/chat_eval.json`. It includes supported questions, unsupported questions that should escalate, and multi-turn follow-ups. It also records stage latencies for query rewrite, embedding, retrieval, reranking, generation, and total response time. The latest measured result is written to `data/evals/chat_benchmark_latest.json`.

`make coverage` measures application-code coverage only. The test suite now lives outside `src/`, so the coverage percentage reflects the product code rather than counting test modules as part of the package.

## Engineering Principles

The codebase aims for strong separation of concerns rather than strict textbook architecture claims:

- API routes handle request/response translation
- application services coordinate the chat and ingestion flows
- infrastructure adapters isolate OpenRouter, Gemini, and Qdrant specifics
- retrieval strategies are swappable behind a small interface

In SOLID terms, the strongest principles reflected here are single responsibility and dependency inversion. The intent is practical: make the system easier to test, reason about, and change.

## Endpoint Contract

`POST /chat`

Request:

```json
{
  "message": "How much is the late payment fee?",
  "history": [
    {
      "role": "user",
      "content": "I forgot to pay my bill."
    }
  ],
  "session_id": "optional-session-id"
}
```

Response:

```json
{
  "reply": "The late payment fee is IDR 50,000 after 14 days overdue.",
  "escalate": false
}
```

## Prompt Design

The system prompt is strict on grounding. It defines the assistant as a telco support agent, forbids guessing, and requires escalation whenever the retrieved evidence is missing or weak. The model is also forced to emit a structured JSON object with `reply`, `escalate`, `confidence`, and `rationale`, which keeps the application logic simple and auditable.

For long conversations, a separate prompt composes the retrieval query from the latest user message plus recent history. This handles references like "that bill" or "that plan" without embedding the full transcript. That rewrite step is intentionally cheaper than final answering: it uses `gpt-oss-20b`, while the higher-stakes grounded answer still uses `Kimi K2.5`.

## OpenRouter Routing

The OpenRouter integration stays on the `Responses API`, but I pass provider preferences through the request body using `extra_body={"provider": ...}` in the Python `openai` SDK.

This lets me separate model responsibilities:

- query rewrite prefers `groq` for `openai/gpt-oss-20b`
- final answer generation prefers `moonshotai/int4` for `moonshotai/kimi-k2.5`
- reranking uses the dedicated OpenRouter rerank endpoint separately

Those provider preferences are configurable through `.env`, so the system can trade off cost, latency, and reliability without changing application code.

## Chunking Strategy

The source corpus is already highly structured, so I chunked one bullet per chunk with zero overlap. That produces 12 atomic chunks. This works better than generic token chunking because each bullet is already a complete policy or procedure statement, and overlap would mostly duplicate noise.

## Retrieval Strategy

The service supports two retrieval modes:

- `dense`: Gemini embeddings + Qdrant vector search
- `hybrid`: dense retrieval plus BM25 lexical retrieval fused with reciprocal rank fusion (RRF)

I implemented hybrid retrieval because exact terms such as `IDR 50,000`, `30 days`, `call 123`, and `5GB hotspot` are common failure points in larger RAG systems.

I expanded the benchmark with harder keyword-heavy and number-heavy cases such as:

- `14 days overdue 50000 late fee`
- `billing errors call 123 free network`
- `pro plan 199000 50gb 5gb hotspot`
- `authorized store valid id sim replacement`

After benchmarking both modes on the expanded 28-case labeled eval set for this assignment corpus, both achieved:

- top-1 accuracy: `1.00`
- top-3 recall: `1.00`

Because the corpus is tiny and highly structured, dense retrieval already saturates the benchmark. For that reason, I keep `dense` as the default and retain `hybrid` as a production-oriented extension that can be enabled with `RETRIEVAL_STRATEGY=hybrid`.

Operationally, hybrid retrieval is also a little more expensive to maintain: the dense index lives in Qdrant and can be updated incrementally, while the in-memory BM25 lexical index must be rebuilt whenever the corpus, tokenization logic, or lexical weighting configuration changes. That is another reason dense-only is the cleaner default for this small assignment corpus.

## Reranking

I also implemented an optional reranking stage using `cohere/rerank-4-fast` through OpenRouter. In a larger B2B knowledge base, this is where I would expect a meaningful precision improvement:

- retrieve a wider candidate pool
- rerank the candidates for semantic relevance
- keep only the best few chunks for the final answer

On this assignment corpus, reranking did not improve retrieval quality:

- `dense`: top-1 `1.00`, top-3 `1.00`, average latency `582.09ms`
- `hybrid_rrf`: top-1 `1.00`, top-3 `1.00`, average latency `565.58ms`
- `dense_rerank`: top-1 `1.00`, top-3 `1.00`, average latency `983.50ms`, `28` search units
- `hybrid_rrf_rerank`: top-1 `1.00`, top-3 `1.00`, average latency `990.03ms`, `28` search units

So reranking is included as an enterprise-oriented extension that belongs in the architecture, but it is not enabled by default because it roughly doubles retrieval latency here without increasing benchmark accuracy.

To enable it for experimentation, set:

- `RERANK_ENABLED=true`
- `RERANK_MODEL=cohere/rerank-4-fast`

## Chat Evaluation

In addition to retrieval-only benchmarking, I added a small end-to-end chat benchmark to verify conversational behavior on real provider calls. The dataset includes:

- supported direct questions
- unsupported questions that must escalate
- multi-turn follow-ups that rely on recent history

The latest measured result in `data/evals/chat_benchmark_latest.json` is:

- cases: `6`
- pass rate: `1.00`
- average query rewrite latency: `1658.43ms`
- average embedding latency: `592.63ms`
- average retrieval latency: `4.54ms`
- average generation latency: `8747.35ms`
- average total latency: `11003.00ms`

This benchmark is intentionally small and assignment-shaped, but it gives the reviewer something concrete to inspect for multi-turn behavior, escalation behavior, and stage-by-stage latency.

## Why This Is Not Overengineered

Some parts of this submission are more polished than the bare minimum, such as the Docker workflow, retrieval benchmark, optional hybrid retriever, and optional reranker. I included them to demonstrate reproducibility and decision-making under realistic production constraints, not to inflate complexity.

Where the data did not justify extra complexity, I explicitly kept the simpler path:

- dense retrieval remains the default because the benchmark showed no lift from hybrid retrieval
- reranking remains optional because the benchmark showed added latency and search-unit cost without accuracy lift on this corpus
- the architecture stays lightweight and avoids external orchestration frameworks
- the API contract remains minimal: `reply` plus `escalate`

## Embedding Model Choice

I chose `gemini-embedding-2-preview` directly from Google because Google documents asymmetric retrieval support for query and document embeddings. In the implementation, I use Gemini's retrieval task types so query and document embeddings are generated with the right retrieval intent:

- query: `task: question answering | query: ...`
- document: `title: ... | text: ...`

Concretely:

- query embeddings use `RETRIEVAL_QUERY`
- document embeddings use `RETRIEVAL_DOCUMENT`
- document title is passed separately to Gemini as additional retrieval context

I also keep the full `3072` dimensions because Google documents that as the highest-quality normalized embedding size for this model family.

## Current Limitation

The current evaluation set is still small and assignment-shaped. It is useful for proving correctness on the provided corpus, but it is not yet a substitute for a larger production-style benchmark with noisier documents, paraphrased customer language, ambiguous follow-ups, and answer-groundedness review.

In production, I would expand the eval set substantially, keep reranking and hybrid retrieval available for larger corpora, and add more rigorous answer-quality checks such as groundedness review and regression gates tied to real support scenarios.
