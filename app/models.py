from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1)
    system_prompt: str = ""
    model: str | None = Field(default=None, min_length=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class ChatResponse(BaseModel):
    response: str
    cached: bool
    similarity: float | None
    latency_ms: float


class CacheStats(BaseModel):
    entries: int
    hits: int
    misses: int
    hit_rate: float
