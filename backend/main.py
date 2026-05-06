from pathlib import Path

env_path = Path(__file__).resolve().parents[1] / ".env"
if env_path.exists():
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=env_path)

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.routes.ask import router as ask_router
from app.routes.ingestion import router as ingestion_router, start_auto_ingestion_worker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("KM Knowledge Agent is Working")


@asynccontextmanager
async def lifespan(app: FastAPI):
    start_auto_ingestion_worker()
    yield

app = FastAPI(
    title="AI Redaction Agent",
    description="RAG-powered system for Q&A and redaction with multi-turn memory",
    version="2.0.0",
    lifespan=lifespan,
)

app.include_router(ask_router)
app.include_router(ingestion_router)


@app.get("/")
def root():
    return {"message": "KM AI Agent is running!"}
