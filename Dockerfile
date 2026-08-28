# Ingestion (fetch FastAPI docs, chunk, embed, index) runs at BUILD time, not
# startup, so the container starts fast and doesn't need network access to
# GitHub or re-download models on every restart. The embedding/reranker
# models download from Hugging Face during this step and get baked into the
# image layer too, so a running container needs no external calls except to
# the Gemini API.
FROM python:3.11-slim

WORKDIR /app

# System deps for sentence-transformers/torch wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY scripts/ scripts/
COPY eval/ eval/

# Ingestion needs no secrets (embeddings/reranking run locally) so it's safe
# to run during the image build, before any runtime env vars are injected.
RUN python -m src.ingestion.run_ingest

EXPOSE 8000

# GOOGLE_API_KEY, API_KEY, and (optionally) LANGFUSE_* are injected by the
# hosting platform's secrets mechanism at runtime — never baked in here.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
