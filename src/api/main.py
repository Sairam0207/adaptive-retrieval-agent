"""FastAPI demo surface: shows the full retrieve/grade/correct/answer trace
per query, so the self-correction behavior is visible, not just the final answer.

Serves a minimal chat UI at "/" and the JSON API at "/query". Both require an
API key (via the X-API-Key header) whenever settings.api_key is set; it's left
open when unset so local development needs no extra setup.
"""
import logging

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from src.agent.graph import ask
from src.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Adaptive Retrieval Agent")


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """No-op when API_KEY isn't configured (local dev). Once set, every
    request to a protected route must echo it back in the X-API-Key header —
    this is what stops a public deployment from being spammed against your
    Gemini quota by anyone who finds the URL."""
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)


class QueryResponse(BaseModel):
    answer: str
    abstained: bool
    retries_used: int
    sources: list[str]
    trace: list[dict]


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
def query(request: QueryRequest) -> QueryResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question must not be blank")

    try:
        result = ask(question)
    except Exception:
        # Anything that escapes here means both the primary and fallback
        # Gemini models failed (or some other unrecoverable error) — surface
        # a clean 503 instead of leaking a stack trace to the client.
        logger.exception("ask() failed for question: %r", question)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The agent is temporarily unavailable. Please try again shortly.",
        )

    return QueryResponse(
        answer=result["answer"],
        abstained=result["abstained"],
        retries_used=result["retry_count"],
        sources=sorted({c.source for c in result.get("retrieved_chunks", [])}),
        trace=result.get("trace", []),
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


_UI_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Adaptive Retrieval Agent</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 720px; margin: 40px auto; padding: 0 16px; }
  textarea { width: 100%; height: 80px; font: inherit; padding: 8px; box-sizing: border-box; }
  button { margin-top: 8px; padding: 8px 16px; font: inherit; cursor: pointer; }
  #answer { white-space: pre-wrap; margin-top: 20px; padding: 12px; border-radius: 6px; background: #f4f4f4; }
  .meta { color: #666; font-size: 0.85em; margin-top: 8px; }
  .abstained { color: #b45309; font-weight: 600; }
</style>
</head>
<body>
  <h2>Adaptive Retrieval Agent</h2>
  <p>Ask a question about FastAPI. The agent grades its own retrieval and abstains rather than guessing.</p>
  <textarea id="q" placeholder="e.g. How do I add CORS to a FastAPI app?"></textarea>
  <br>
  <button id="ask">Ask</button>
  <div id="answer" style="display:none"></div>

  <script>
    const KEY_STORAGE = "ara_api_key";

    async function ask() {
      const question = document.getElementById("q").value.trim();
      if (!question) return;

      const answerBox = document.getElementById("answer");
      answerBox.style.display = "block";
      answerBox.textContent = "Thinking...";

      const headers = { "Content-Type": "application/json" };
      let apiKey = localStorage.getItem(KEY_STORAGE);
      if (apiKey) headers["X-API-Key"] = apiKey;

      let resp = await fetch("/query", { method: "POST", headers, body: JSON.stringify({ question }) });

      if (resp.status === 401) {
        apiKey = prompt("This deployment requires an API key:");
        if (!apiKey) { answerBox.textContent = "Cancelled — API key required."; return; }
        localStorage.setItem(KEY_STORAGE, apiKey);
        headers["X-API-Key"] = apiKey;
        resp = await fetch("/query", { method: "POST", headers, body: JSON.stringify({ question }) });
      }

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        answerBox.textContent = "Error: " + (err.detail || resp.statusText);
        return;
      }

      const data = await resp.json();
      const abstainNote = data.abstained ? '<div class="abstained">The agent abstained rather than guess.</div>' : "";
      answerBox.innerHTML =
        data.answer.replace(/</g, "&lt;") +
        abstainNote +
        '<div class="meta">retries: ' + data.retries_used + " | sources: " + (data.sources.join(", ") || "none") + "</div>";
    }

    document.getElementById("ask").addEventListener("click", ask);
    document.getElementById("q").addEventListener("keydown", (e) => {
      if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) ask();
    });
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def ui() -> str:
    return _UI_HTML
