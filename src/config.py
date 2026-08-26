from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    google_api_key: str = ""

    grader_model: str = "gemini-flash-lite-latest"
    generator_model: str = "gemini-flash-latest"

    qdrant_path: str = "./data/qdrant_local"
    qdrant_url: str = ""
    qdrant_collection: str = "ara_docs"

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    max_retries: int = 2
    top_k_retrieve: int = 20
    top_k_rerank: int = 5

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    bm25_index_path: str = "./data/bm25_index.pkl"
    raw_docs_path: str = "./data/raw"


settings = Settings()
