# Question 2: Production Design and Evaluation

Artifacts:

- Diagram PNG: [architecture.png](architecture.png)
- Diagram SVG: [architecture.svg](architecture.svg)
- Mermaid source: [architecture.mmd](architecture.mmd)

## Architecture Summary

The production system uses one shared orchestration core behind two channel adapters: chat for web/mobile and voice for inbound SIP calls. Both channels converge into the same conversation orchestration layer so retrieval, escalation logic, prompt policy, and observability stay consistent across interfaces.

## Executive Summary

This design aims to balance safety, operational simplicity, and realistic production growth:

- one orchestration core shared by chat and voice
- a versioned RAG pipeline that can be updated without redeploying the app
- Redis-backed short-term memory so the serving tier stays stateless
- explicit human escalation with transcript and context handoff
- measurable release gates before launch and trace-based monitoring after launch

The core design choice is to treat the system as an orchestration problem, not just a model-call problem. In other words, the production value does not come only from "calling an LLM," but from how session state, retrieval, telephony, evaluation, and escalation are coordinated around that model. That is especially important for a telco support assistant, where billing mistakes, bad handoffs, and poor latency all create real customer risk.

Main components:

- Channel adapters: web/mobile chat requests and SIP/telephony calls.
- Voice gateway: SIP provider or SBC, streaming speech-to-text, and text-to-speech.
- Orchestrator API: session-aware application layer that manages retrieval, prompting, escalation, and model calls.
- Session memory: Redis for short-lived conversational context keyed by `session_id` or `call_id`.
- Knowledge pipeline: document storage, chunking, embeddings, Qdrant indexing, and versioned alias promotion.
- Persistent data: Postgres for transcripts, audit events, escalation metadata, and evaluation datasets.
- Observability: Langfuse traces, OpenTelemetry spans, and infrastructure metrics.

## Main Components and Responsibilities

### Channel Adapters

The channel adapters normalize input from different surfaces into one shared conversation contract. For chat, that means accepting web or mobile messages, validating the request, and forwarding the session plus the latest user turn to the orchestrator. For voice, the adapter is more complex: it manages the media session, receives audio frames, handles streaming STT and TTS, and decides when partial speech has become a committed user turn.

### Orchestrator API

The orchestrator is the application brain of the system. It loads the current session memory, decides when retrieval is needed, builds the prompt, calls the model, evaluates whether the answer is safe enough to return, and records the result for observability and audit. This is also the layer that standardizes escalation behavior across chat and voice so the same support policy is enforced in both channels.

### Knowledge Pipeline

The knowledge pipeline is intentionally separated from the serving path. Its job is to ingest new documents, chunk and embed them, write them into a versioned Qdrant collection, run validation checks, and promote a new alias only when retrieval quality is acceptable. This separation keeps knowledge updates operationally safe because a bad index can be rolled back without redeploying or interrupting the serving stack.

### Session Memory and Durable Data

Redis stores short-lived conversation context that needs to be read and updated on every turn with very low latency. Postgres stores durable business records such as transcripts, escalation reasons, and evaluation artifacts. This split keeps the serving tier fast while still giving operations, analytics, and human-support teams a reliable historical record.

### Human Escalation and Agent Desktop

The escalation path is treated as part of the product, not as an afterthought. When the AI cannot answer confidently, the live-agent system should receive enough structured context to continue the conversation immediately: the transcript, the retrieved evidence, the model reply, the escalation reason, and customer identifiers. That reduces repeated questioning and makes the AI feel like a useful front-line assistant rather than a dead-end bot.

## How Voice Differs from Chat

Chat is plain request/response text. Voice adds a real-time media path and stricter latency constraints:

- Incoming audio arrives via SIP/telephony, then gets transcribed by streaming STT.
- Turn-taking matters because callers interrupt, pause, and resume.
- The assistant response must be converted back to speech with TTS.
- Voice flows need barge-in handling, partial transcription handling, and stronger timeout/fallback logic.

The core agent still uses the same retrieval and escalation policy, but the voice adapter is responsible for streaming audio, handling partial utterances, and deciding when to commit a user turn to the orchestrator.

## Why I Prefer a Cascaded Voice Stack

For a telco support assistant, I would treat the default production design as a cascaded pipeline: streaming STT, orchestration plus retrieval, then TTS.

This is not the only valid architecture. A second option is to use a native speech-to-speech model such as OpenAI Realtime or Gemini Live, where audio understanding, reasoning, tool use, and audio generation happen in one stateful session. That approach can reduce latency and improve turn-taking.

I would still default to the cascaded stack here because it is easier to operate safely in a customer-support environment:

- transcripts are explicit and easy to audit
- retrieval grounding is easier to inspect turn by turn
- PII redaction can happen before model or observability export
- each stage can be swapped independently for cost, language, or compliance reasons
- evaluation is simpler because STT quality, retrieval quality, and answer quality can be measured separately

In practice, I would keep native audio models as a second architecture to evaluate for premium voice experiences, but I would start the telco support system with the more controllable cascaded design.

## Knowledge Base Update Flow

Knowledge updates should not require redeployment. The update path is:

1. New support documents land in object storage or an internal admin upload service.
2. An ingestion worker parses, chunks, normalizes metadata, and embeds the new content.
3. The worker writes vectors into a new versioned Qdrant collection.
4. A validation step runs retrieval spot checks against a known query set.
5. If the checks pass, the serving alias moves to the new collection.
6. If retrieval quality regresses, the alias rolls back to the previous collection immediately.

This separates application deploys from knowledge updates and makes rollback operationally simple.

## Conversation Memory

Short-term session memory lives in Redis so the app tier stays stateless and can scale horizontally. The orchestrator reads the session state at the beginning of each turn, appends the latest interaction, and writes back the updated memory after the response.

Longer-lived records such as transcripts, escalation reasons, and human handoff notes go to Postgres. This gives the business a durable audit trail and allows offline evaluation jobs to analyze real traffic.

## Human Escalation

Escalation should hand off more than a boolean. The live-agent system should receive:

- full transcript or the latest call/chat summary
- customer and session identifiers
- retrieved chunks used by the AI
- the model reply
- the explicit reason for escalation

This reduces duplicate questioning and makes the handoff feel seamless.

## Security and Privacy

Because this is a B2B customer-support system, I would make transport security and data-handling constraints explicit in the architecture:

- Voice transport should use TLS for SIP signaling and SRTP for media whenever the telephony provider supports secure media.
- Audio streaming between the telephony layer and the voice adapter should use secure WebSockets.
- Prompt and trace payloads should be redacted or tokenized before being sent to external observability systems.
- Model routing should prefer zero-data-retention or no-data-collection endpoints when customer-sensitive traffic is involved.
- Access to transcripts, retrieved chunks, and escalation payloads should be scoped by role and audit-logged.

These controls matter because the system handles billing conversations, account context, and potentially personally identifiable information.

## Scalability Concern

The first scalability bottleneck is likely the combination of vector search load and model latency during peak support hours. I would address this by:

- keeping app servers stateless and scaling them horizontally
- externalizing session memory in Redis
- caching repeated retrieval results for common intents
- splitting ingestion from serving so re-index jobs do not contend with live traffic
- rate-limiting or queueing lower-priority workloads such as batch evaluation

I would avoid sticky sessions unless a telephony provider requires them at the edge. Externalized session state is more flexible.

## Before Launch: Evaluation Strategy

Before launch, I would evaluate the system the way I would evaluate a high-risk support feature: with a small but carefully labeled dataset, clear quality thresholds, and explicit failure gates. Because this assistant answers billing and plan questions, I care more about groundedness and safe escalation than stylistic fluency.

### Test Dataset

I would build a labeled evaluation set with roughly 60 to 80 examples from the provided domains:

- direct factual billing questions
- plan comparison questions
- troubleshooting questions
- paraphrases and multilingual variants
- unsupported questions that must escalate
- multi-turn follow-ups with references such as "that bill" or "that plan"
- adversarial prompts trying to induce hallucinations

Each example would include:

- expected escalation outcome
- expected supporting document or chunk
- expected answer points, not just one exact string

### Metrics

- Answer accuracy: whether the response contains the required facts.
- Retrieval top-k recall: whether the correct supporting chunk appears in the top results.
- Unsupported-query escalation recall: whether out-of-scope questions escalate instead of being answered.
- Hallucination rate: percentage of answers that introduce unsupported claims.
- Latency: p50 and p95 response time.

### Review Method

I would use a hybrid evaluation approach:

- manual review for the golden dataset because the domain is small and policy-sensitive
- automated checks for schema validity, retrieval hit rate, and escalation correctness
- LLM-as-judge for groundedness and answer quality on larger samples, but never as the only release gate

### Release Thresholds

- supported-answer accuracy at least 90%
- retrieval top-3 recall at least 95%
- unsupported-query escalation recall at 100%
- hallucination rate at or below 2%
- p95 chat latency below 2.5 seconds
- for voice, p95 time-to-first-audio should stay below 1.2 seconds and barge-in handling should succeed reliably enough that interrupted bot speech is cut off almost immediately in user testing

## In Production: Monitoring and Observability

Once the assistant is live, I would monitor both traditional service health and AI-specific behavior. Uptime alone is not enough for this kind of system; a voice or chat agent can be fully available while still being operationally bad because it is hallucinating, retrieving the wrong chunks, or escalating too often.

### Metrics

- Escalation rate by intent and release version
  Detects whether the assistant suddenly becomes too conservative or too risky.
- Retrieval empty-hit rate and top-score distribution
  Detects ingestion or embedding problems before users report them.
- Groundedness or hallucination score on sampled traces
  Detects answer-quality regressions that simple uptime monitoring misses.
- p50 and p95 latency by stage
  Separates model slowness from retrieval slowness or voice-processing slowness.
- Token and cost per request
  Protects unit economics and catches runaway prompt growth.

### Tooling

- Langfuse for trace-level observability, dataset experiments, and online evaluations.
- OpenTelemetry for vendor-neutral spans across the app, retrieval, and model calls.
- Prometheus/Grafana or the company standard for infrastructure metrics and alerts.
- Structured logs shipped to the company log platform for auditability.

### Detecting a Quality Drop

If answer quality drops suddenly, I would:

1. Inspect Langfuse traces for the failing segment: retrieval, prompt version, or model version.
2. Compare retrieval hit rates and groundedness scores against the previous baseline.
3. Roll back the prompt, model config, or vector alias depending on the failing layer.
4. Increase escalation conservatism temporarily if user risk is high.

## Failure Mode Analysis

My default production posture is to treat these issues as incident-response problems, not just model-quality problems. For each major failure mode, I would follow the same sequence:

1. contain user risk first
2. identify the failing layer with traces and metrics
3. roll back the smallest possible surface area
4. validate on a targeted offline slice before restoring normal autonomy

### Scenario 1: The LLM starts hallucinating billing policies

Response plan:

- immediately move billing intents into a safer operating mode by increasing escalation behavior and reducing the range of questions the assistant can answer autonomously
- inspect recent Langfuse traces to separate three possibilities: prompt drift, model drift, or retrieval supplying weak evidence
- compare the active prompt, model, and provider configuration against the last known good release
- sample the failing traces and verify whether the retrieved chunks actually support the answer text
- if the problem is generation drift, roll back the prompt or model configuration first rather than touching the retrieval layer
- if customer risk is material, temporarily force all billing-policy questions to escalate until the issue is contained
- run the billing-focused offline eval suite before restoring the previous answer autonomy
- keep an audit record of the incident window so support and operations teams can review affected conversations if needed

### Scenario 2: A knowledge-base update causes irrelevant retrieval

Response plan:

- roll back the serving alias to the previous Qdrant collection immediately so the live system returns to the last known good index
- compare document counts, chunk counts, metadata coverage, and embedding job logs between the bad and good index versions
- inspect top-score distributions, empty-hit rate, and validation-set retrieval hit rates to identify whether the regression came from chunking, metadata loss, or embedding quality
- confirm whether the issue is corpus-related or query-side by replaying a small set of failing production traces against both index versions
- pause further promotions from the ingestion pipeline until validation is fixed
- re-run ingestion validation before promoting a new collection again

This is why the index should be versioned and promoted via alias instead of overwritten in place.
