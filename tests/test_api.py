from fastapi.testclient import TestClient

import pytest
from redis.exceptions import ConnectionError

from app import main
from app.embeddings import EmbeddingServiceError


@pytest.fixture
def client(monkeypatch, redis_client, namespace, embeddings, llm):
    # General infrastructure cases run without the demo's domain restriction.
    monkeypatch.setattr(main, "SupportFAQPolicy", lambda: None)
    monkeypatch.setattr(main, "create_redis_client", lambda: redis_client)
    monkeypatch.setattr(main, "RemoteEmbeddings", lambda url: embeddings)
    monkeypatch.setenv("CACHE_NAMESPACE", namespace)
    with TestClient(main.app) as client:
        yield client


def test_first_request_misses_and_paraphrase_hits(client, llm) -> None:
    first = client.post("/v1/chat", json={"prompt": "What is the capital of France?"})
    second = client.post("/v1/chat", json={"prompt": "Tell me the capital city of France."})

    assert first.status_code == 200
    assert first.json()["cached"] is False
    assert second.status_code == 200
    assert second.json()["cached"] is True
    assert len(llm) == 1
    assert first.json()["response"] == second.json()["response"]


def test_stats_and_clear_cache(client) -> None:
    client.post("/v1/chat", json={"prompt": "How do I reset my password?"})

    stats = client.get("/v1/cache/stats")
    assert stats.status_code == 200
    assert stats.json()["entries"] == 1

    cleared = client.delete("/v1/cache")
    assert cleared.status_code == 204
    assert client.get("/v1/cache/stats").json()["entries"] == 0


def test_api_restart_preserves_cache(client):
    payload = {"prompt": "A persistent question"}
    assert client.post("/v1/chat", json=payload).json()["cached"] is False
    # A new lifespan creates a new client/cache lifecycle, using the same Redis data.
    with TestClient(main.app) as restarted:
        assert restarted.post("/v1/chat", json=payload).json()["cached"] is True


def test_redis_outage_returns_503(client, monkeypatch):
    def unavailable(*args, **kwargs):
        raise ConnectionError("offline")

    monkeypatch.setattr(main.app.state.cache.redis, "hget", unavailable)
    response = client.post("/v1/chat", json={"prompt": "hello"})
    assert response.status_code == 503
    assert response.json() == {"detail": "Cache backend unavailable"}


def test_embedding_outage_returns_503_without_storing(client, monkeypatch):
    def unavailable(text):
        raise EmbeddingServiceError("offline")

    monkeypatch.setattr(main.app.state.cache.embeddings, "encode", unavailable)
    response = client.post("/v1/chat", json={"prompt": "A new question"})
    assert response.status_code == 503
    assert main.app.state.cache.stats()["entries"] == 0


def test_llm_failure_is_not_cached(client, monkeypatch):
    from app.llm import LLMError

    def unavailable(*args):
        raise LLMError("OpenAI is unavailable")

    monkeypatch.setattr(main.app.state.llm, "generate", unavailable)
    response = client.post("/v1/chat", json={"prompt": "An uncached question"})
    assert response.status_code == 502
    assert main.app.state.cache.stats()["entries"] == 0


def test_effective_model_is_used_for_generation_and_isolation(client, llm):
    import json
    from app import main

    first = client.post("/v1/chat", json={"prompt": "Question"})
    explicit = client.post("/v1/chat", json={"prompt": "Question", "model": "test-model"})
    different = client.post("/v1/chat", json={"prompt": "Question", "model": "another-model"})
    assert first.json()["cached"] is False
    assert explicit.json()["cached"] is True
    assert different.json()["cached"] is False
    assert [json.loads(call.content)["model"] for call in llm] == ["test-model", "another-model"]
    assert main.app.state.cache.stats()["entries"] == 2


def test_missing_key_can_start_but_cannot_generate(client):
    from app import main
    main.app.state.llm.configured = False
    result = client.post("/v1/chat", json={"prompt": "A new question"})
    assert result.status_code == 503
    assert "OPENAI_API_KEY" in result.json()["detail"]
    assert main.app.state.cache.stats()["entries"] == 0


@pytest.mark.parametrize("prompt", ["Where is my delivery?", "Track order 1234", "Explain Python decorators"])
def test_support_scope_bypasses_cache_but_calls_llm(client, llm, prompt):
    from app.support_policy import SupportFAQPolicy
    main.app.state.cache.policy = SupportFAQPolicy()
    for _ in range(2):
        result = client.post("/v1/chat", json={"prompt": prompt})
        assert result.status_code == 200
        assert result.json()["cached"] is False
    assert len(llm) == 2
    assert main.app.state.cache.stats()["entries"] == 0
