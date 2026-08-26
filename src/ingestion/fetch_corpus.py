"""Downloads a real, verifiable open-source documentation corpus (FastAPI docs)
into data/raw/. Swap FASTAPI_DOC_PATHS or point src/config.py's raw_docs_path
at any folder of .md files to use a different corpus.
"""
import os
import sys

import requests

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import settings

BASE_URL = "https://raw.githubusercontent.com/fastapi/fastapi/master/docs/en/docs/"

FASTAPI_DOC_PATHS = [
    "index.md",
    "async.md",
    "alternatives.md",
    "history-design-future.md",
    "benchmarks.md",
    "tutorial/first-steps.md",
    "tutorial/path-params.md",
    "tutorial/query-params.md",
    "tutorial/query-params-str-validations.md",
    "tutorial/body.md",
    "tutorial/cookie-params.md",
    "tutorial/header-params.md",
    "tutorial/response-model.md",
    "tutorial/extra-models.md",
    "tutorial/path-operation-configuration.md",
    "tutorial/request-forms.md",
    "tutorial/handling-errors.md",
    "tutorial/dependencies/index.md",
    "tutorial/security/index.md",
    "tutorial/security/oauth2-jwt.md",
    "tutorial/middleware.md",
    "tutorial/cors.md",
    "tutorial/background-tasks.md",
    "tutorial/testing.md",
    "tutorial/bigger-applications.md",
    "tutorial/sql-databases.md",
    "deployment/index.md",
    "deployment/concepts.md",
    "advanced/index.md",
    "advanced/websockets.md",
]


def fetch_all() -> None:
    os.makedirs(settings.raw_docs_path, exist_ok=True)
    ok, failed = 0, []

    for rel_path in FASTAPI_DOC_PATHS:
        url = BASE_URL + rel_path
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
        except requests.RequestException as exc:
            failed.append((rel_path, str(exc)))
            continue

        flat_name = rel_path.replace("/", "__")
        out_path = os.path.join(settings.raw_docs_path, flat_name)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(resp.text)
        ok += 1

    print(f"Fetched {ok}/{len(FASTAPI_DOC_PATHS)} documents into {settings.raw_docs_path}")
    if failed:
        print("Failed:")
        for path, err in failed:
            print(f"  {path}: {err}")


if __name__ == "__main__":
    fetch_all()
