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
retrieve/grade/correct trace.

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
comparison. The key number to watch: hallucination rate on the unanswerable
subset — naive RAG tends to fabricate answers there; the corrective agent
should abstain instead.

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
