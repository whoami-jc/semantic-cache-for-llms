from contextlib import asynccontextmanager
import os
from time import perf_counter

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from redis.exceptions import RedisError

from app.redis_connection import create_redis_client

from app.cache import SemanticCache
from app.embeddings import RemoteEmbeddings, EmbeddingServiceError
from app.models import CacheStats, ChatRequest, ChatResponse
from app.llm import create_llm, LLMError
from app.settings import Settings
from app.support_policy import SupportFAQPolicy, RETRIEVAL_THRESHOLD

@asynccontextmanager
async def lifespan(app: FastAPI):
    client = create_redis_client()
    embeddings = None
    llm = None
    try:
        client.ping()
        embeddings = RemoteEmbeddings(os.environ.get("EMBEDDINGS_URL", "http://127.0.0.1:8001"))
        app.state.cache = SemanticCache(
            client, embeddings, ttl_seconds=int(os.environ.get("CACHE_TTL_SECONDS", "3600")),
            namespace=os.environ.get("CACHE_NAMESPACE", "semantic-cache"),
            threshold=RETRIEVAL_THRESHOLD,
            policy=SupportFAQPolicy(),
        )
        llm = create_llm(Settings())
        app.state.llm = llm
        yield
    finally:
        if llm is not None:
            llm.close()
        if embeddings is not None:
            embeddings.close()
        client.close()


app = FastAPI(title="Semantic Cache for LLMs", version="0.1.0", lifespan=lifespan)


@app.exception_handler(RedisError)
async def redis_error_handler(request: Request, exc: RedisError):
    return JSONResponse(status_code=503, content={"detail": "Cache backend unavailable"})


@app.exception_handler(EmbeddingServiceError)
async def embedding_error_handler(request: Request, exc: EmbeddingServiceError):
    return JSONResponse(status_code=503, content={"detail": "Embedding service unavailable or incompatible"})


@app.exception_handler(LLMError)
async def llm_error_handler(request: Request, exc: LLMError):
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})


@app.post("/v1/chat", response_model=ChatResponse)
def chat(request: ChatRequest, http_request: Request) -> ChatResponse:
    started = perf_counter()
    cache = http_request.app.state.cache
    llm = http_request.app.state.llm
    model = request.model or llm.default_model
    cache_model = llm.cache_model(model)
    response, similarity, _ = cache.lookup(
        prompt=request.prompt,
        system_prompt=request.system_prompt,
        model=cache_model,
        temperature=request.temperature,
        threshold=request.threshold,
    )
    if response is not None:
        return ChatResponse(
            response=response,
            cached=True,
            similarity=similarity,
            latency_ms=round((perf_counter() - started) * 1000, 2),
        )

    response = llm.generate(request.prompt, request.system_prompt, model, request.temperature)
    cache.store(
        prompt=request.prompt,
        system_prompt=request.system_prompt,
        model=cache_model,
        temperature=request.temperature,
        response=response,
    )
    return ChatResponse(
        response=response,
        cached=False,
        similarity=similarity,
        latency_ms=round((perf_counter() - started) * 1000, 2),
    )


@app.get("/v1/cache/stats", response_model=CacheStats)
def cache_stats(request: Request) -> CacheStats:
    return CacheStats(**request.app.state.cache.stats())


@app.delete("/v1/cache", status_code=204)
def clear_cache(request: Request) -> None:
    request.app.state.cache.clear()
