[![CI](https://github.com/Sairam0207/RAG-Proj/actions/workflows/ci.yml/badge.svg)](https://github.com/Sairam0207/RAG-Proj/actions/workflows/ci.yml)

# Adaptive Retrieval Agent (ARA)

A RAG system that grades its own retrieval before answering, self-corrects
(query reformulation + re-retrieval) when confidence is low, and explicitly
abstains instead of hallucinating when it still can't find enough context.
Exposed as an MCP server, fully traced, and benchmarked against a naive-RAG
baseline with an LLM-as-judge eval harness.

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
python -m eval.run_eval
```

Runs the naive baseline and the corrective agent against `eval/golden_dataset.json`
(20 questions: easy, multi-hop, and deliberately unanswerable), scores both
with an LLM judge (fact coverage, faithfulness, hallucination), and prints a
comparison.

### Results (n=20, gemini-flash-lite-latest as judge)

|                          | naive baseline | corrective agent |
|--------------------------|:--------------:|:-----------------:|
| fact coverage (answerable, n=18) | 0.630 | 0.648 |
| faithfulness (1-5)       | 5.0            | 5.0                |
| hallucination rate       | 0%             | 0%                 |
| correct abstention on unanswerable (n=2) | 0/2 | 2/2 |

Naive RAG and the corrective agent are statistically tied on questions the
corpus can actually answer. The difference shows up on the 2 deliberately
unanswerable questions: naive RAG confidently states a plausible-sounding
answer anyway (e.g. asserting a specific rate-limiting behavior FastAPI
doesn't have), while the corrective agent grades its retrieved context as
insufficient and abstains instead of guessing.

**A note on the metric itself:** naive's answers to the unanswerable
questions score *well* on raw fact-coverage, because the judge only checks
whether the answer states the expected fact — not whether that fact is
actually grounded in the retrieved context. Naive gets there by falling back
on the LLM's own pretrained knowledge, not the corpus. That's a real failure
mode a fact-coverage-only metric rewards and a groundedness-blind eval would
miss — which is why this report scores hallucination/abstention separately
for the unanswerable subset instead of blending it into one average.

The agent isn't free: it retries up to 2x before abstaining, and on this
run it abstained on 4 of the 18 genuinely answerable questions too (an
over-conservative tradeoff from the current grading prompt, not something
this eval currently tunes for).

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
eval/           golden dataset + naive-vs-agent comparison harness
scripts/        interactive CLI demo
```
