[![CI](https://github.com/Sairam0207/RAG-Proj/actions/workflows/ci.yml/badge.svg)](https://github.com/Sairam0207/RAG-Proj/actions/workflows/ci.yml)

# Adaptive Retrieval Agent (ARA)

A RAG system that grades its own retrieval before answering, self-corrects
(query reformulation + re-retrieval) when confidence is low, and explicitly
abstains instead of hallucinating when it still can't find enough context.
Exposed as an MCP server, fully traced, and benchmarked against two naive-RAG
baselines with both a hand-rolled LLM-as-judge harness and RAGAS.

Corpus: real FastAPI documentation (fetched live from GitHub), so retrieval
quality is independently verifiable against the source.

## Architecture

```
query -> hybrid retrieve (dense + BM25) -> rerank -> GRADE
                                                        |
                              sufficient? --------------+------- insufficient?
                                  |                              |
                              generate                    reformulate query
                                  |                              |
                               answer                    retry (bounded) -> retrieve
                                                                  |
                                                     still insufficient? -> abstain
```

See `src/agent/graph.py` for the LangGraph state machine and `src/agent/nodes.py`
for each step's implementation.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
pip install -r requirements-eval.txt   # only needed to run the RAGAS eval

cp .env.example .env            # then set GOOGLE_API_KEY (free tier: https://aistudio.google.com/apikey)
```

No Docker or external services required by default — Qdrant runs embedded
(on-disk), embeddings/reranking run locally via sentence-transformers, and
observability falls back to console logging if Langfuse keys aren't set.
For full nested traces (retrieve/grade/reformulate/generate spans per query),
create a free project at https://cloud.langfuse.com and set
`LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` in `.env`.

## Ingest the corpus

```bash
python -m src.ingestion.run_ingest
```

Fetches FastAPI's docs into `data/raw/` (skipped if already present), chunks
them, and builds both the dense (Qdrant) and sparse (BM25) indexes.

To use a different corpus: drop `.md` files into `data/raw/` yourself and
skip the fetch, or edit `FASTAPI_DOC_PATHS` in `src/ingestion/fetch_corpus.py`.

## Run the interactive demo

```bash
python scripts/demo.py
```

Try a question that should retrieve cleanly (e.g. "How do I add CORS to a
FastAPI app?"), then one designed to force a retry or abstain (e.g. "How do
I configure FastAPI's built-in rate limiting?" — FastAPI has none, so the
agent should say so rather than inventing an answer).

## Run the API server

```bash
uvicorn src.api.main:app --reload
```

`POST /query {"question": "..."}` returns the answer plus the full
retrieve/grade/correct trace. A minimal chat UI is served at `/`.

Set `API_KEY` in `.env` to require an `X-API-Key` header on `/query` — left
blank by default so local dev needs no extra setup, but **must** be set
before deploying anywhere publicly reachable (an open endpoint is an open
invitation to burn through your Gemini quota). The bundled UI prompts for the
key once and remembers it in the browser's local storage.

## Run with Docker

```bash
docker build -t ara .
docker run -p 8000:8000 -e GOOGLE_API_KEY=... -e API_KEY=... ara
```

The corpus is fetched, chunked, embedded, and indexed **during the image
build**, not at container startup — so the container starts instantly and
needs no network access beyond the Gemini API at runtime. Pass
`LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` as additional
`-e` flags for full tracing; omit them and it falls back to console logs.
Secrets are injected as environment variables at deploy time (via your
platform's secrets manager) — `.env` is never copied into the image.

## Run the MCP server

```bash
python -m src.mcp_server.server
```

Point an MCP client (Claude Desktop, Claude Code) at this server to call
`query_knowledge_base` as a tool from another agent.

## Run the evaluation

```bash
python -m eval.run_eval      # hand-rolled LLM judge, three arms
python -m eval.ragas_eval    # RAGAS metrics over the same answers
```

`eval/golden_dataset.json` holds 28 hand-written questions: 14 easy, 4
multi-hop, and 10 **unanswerable**, split into two deliberately different
kinds:

- `nonexistent_feature` (7) — asks about FastAPI features that do not exist
  (built-in rate limiting, an ORM, an admin dashboard, a cron scheduler).
  Any confident answer here is a fabrication.
- `outside_corpus` (3) — asks about *real* FastAPI features
  (`StaticFiles`, `StreamingResponse`, `lifespan`) that were deliberately
  left out of the indexed corpus. A confident answer here may be factually
  true and still ungrounded, because it cannot have come from retrieval.
  Only abstention is correct.

That second category is the one that matters. It separates "did the model say
something false" from "did the model answer from the corpus", and those are
not the same question.

### The bug that invalidated the first version of this eval

The original comparison claimed the corrective agent beat naive RAG on
hallucination. It did not. `naive_baseline.py` imported
`GENERATE_SYSTEM_PROMPT` from the agent — the same prompt, ending with *"If
the context does not fully support an answer, say so explicitly rather than
guessing."* The "naive" baseline had been handed the exact intervention it
was supposed to be a control for, so both arms shared the treatment and the
measured difference was never attributable to the correction loop.

Expanding the unanswerable set from 2 to 10 is what surfaced it: the judge
reported **0 hallucinations for every arm on every question**, including
questions with no answer in the corpus. A metric that never fires is not
evidence of a good system, it is evidence of a broken measurement.

The fix is a third arm. `naive_answer(question, grounded=False)` uses a
genuinely naive prompt with no groundedness instruction, so the eval can now
attribute results to the right cause:

| arm | groundedness prompt | grade/retry/abstain loop |
|-----|:---:|:---:|
| `naive_ungrounded` | no | no |
| `naive_grounded`   | yes | no |
| `agent`            | yes | yes |

### Results (n=28, gemini-flash-lite-latest as judge)

|                                        | naive_grounded | agent |
|----------------------------------------|:--------------:|:-----:|
| fact coverage (answerable, n=18)       | 0.630          | 0.648 |
| faithfulness (1–5)                     | 5.0            | 5.0   |
| judge-flagged hallucinations           | 0/28           | 0/28  |
| abstained on unanswerable (n=10)       | 0/10           | 10/10 |
| — of which `nonexistent_feature` (n=7) | 0/7            | 7/7   |
| — of which `outside_corpus` (n=3)      | 0/3            | 3/3   |
| abstained on answerable (n=18)         | 0/18           | 4/18  |
| avg retries per question               | 0              | 1.21  |

Read honestly, this is not a clean win:

- The agent abstains correctly on **10/10** unanswerable questions, including
  all 3 where the true answer exists but is outside the corpus. That is the
  behaviour the whole state machine is for, and it works.
- It also abstains on **4/18** questions the corpus *can* answer. That is a
  22% false-abstention rate, and it is a real cost, not a rounding error.
- `naive_grounded` never abstains — it has no abstain path — but it also
  never got flagged as hallucinating, because the groundedness sentence in
  its prompt already makes it decline in prose. **On this corpus, one line of
  prompting recovers most of what the loop provides, at 1/2.2 the LLM calls.**

That last point is the most useful thing this eval produced, and it argues
*against* the architecture it was built to showcase.

### Why RAGAS was added

The hand-rolled judge asks one binary question — "did this hallucinate?" —
and it answered "no" 84 times out of 84. It cannot rank systems it never
separates. RAGAS scores groundedness continuously instead:

| metric | what it catches that the binary flag missed |
|--------|---------------------------------------------|
| `faithfulness` | decomposes an answer into claims and checks each against retrieved context — partial ungroundedness shows up as 0.6, not "no" |
| `answer_relevancy` | whether the answer engages the question at all; this is what penalises abstention |
| `context_precision` | how much of what the retriever returned was actually relevant |
| `context_recall` | whether retrieval found everything the reference needs |

`faithfulness` and `answer_relevancy` deliberately pull against each other.
Abstaining scores ~1.0 on faithfulness (no claims, nothing to contradict) and
~0 on relevancy. Reporting either alone would flatter this system, so
`eval/ragas_eval.py` scores answered and abstained rows as separate
populations and prints the abstention rate beside them rather than blending
them into one average.

Two implementation notes worth knowing before you run it:

- `ResponseRelevancy` must be constructed with `strictness=1`. Its default of
  3 requests three candidates in a single call, which Gemini rejects with
  `400 INVALID_ARGUMENT: Multiple candidates is not enabled for this model` —
  and RAGAS swallows that and silently reports `nan` instead of failing.
- RAGAS runs with `RunConfig(max_workers=1)`. Its default fan-out trips the
  free-tier per-minute quota within seconds and poisons a whole batch of
  scores with 429s.

Generation is checkpointed per sample to `eval/ragas_samples.json`, so a run
killed by a daily quota wall resumes where it stopped:

```bash
python -m eval.ragas_eval --limit 4 --arms agent   # smoke test
python -m eval.ragas_eval --score-only             # re-score cached answers
```

### Known limitations

- The three-arm RAGAS comparison across all 28 questions has not been run to
  completion; the free-tier daily quota on `gemini-flash-latest` is the
  binding constraint, and partial numbers are not reported here as if they
  were final. `--score-only` reproduces whatever is cached.
- Single judge model, and it is the same family as the generator. A judge
  sharing the generator's blind spots is a known weakness of LLM-as-judge
  setups and is not controlled for here.
- 28 questions is small. It is enough to have caught a real methodology bug;
  it is not enough for a confident claim about effect size.

## Tests

```bash
pytest
```

## Project layout

```
src/
  ingestion/    chunking, embedding, indexing, corpus fetch
  retrieval/    hybrid search (dense+BM25) + cross-encoder reranking
  agent/        LangGraph state machine: retrieve/grade/reformulate/generate/abstain
  mcp_server/   exposes the agent as an MCP tool
  observability/ Langfuse tracing with console-log fallback
  api/          FastAPI demo endpoint
eval/           golden dataset, LLM-judge harness, RAGAS scorer
scripts/        interactive CLI demo
```
