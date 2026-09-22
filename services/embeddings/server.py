from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import torch

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


@asynccontextmanager
async def lifespan(app):
    torch.set_num_threads(2)
    app.state.model = SentenceTransformer(MODEL, revision=REVISION, device="cpu")
    yield


app = FastAPI(title="Local English Embeddings", lifespan=lifespan)


class EmbeddingRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=64)


@app.get("/health")
def health():
    return {"model": MODEL, "revision": REVISION, "dimension": app.state.model.get_sentence_embedding_dimension()}


@app.post("/embed")
def embed(request: EmbeddingRequest):
    vectors = app.state.model.encode(request.texts, normalize_embeddings=True)
    return {**health(), "embeddings": vectors.tolist()}





