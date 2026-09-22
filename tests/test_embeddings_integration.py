import os

import numpy as np
from fastapi.testclient import TestClient

from app import main
from app.cache import SemanticCache
from app.embeddings import RemoteEmbeddings


def test_docker_embeddings_through_api_and_redis(monkeypatch, redis_client, namespace, llm):
    monkeypatch.setattr(main, "SupportFAQPolicy", lambda: None)
    monkeypatch.setattr(main, "create_redis_client", lambda: redis_client)
    monkeypatch.setenv("CACHE_NAMESPACE", namespace)
    monkeypatch.setenv("EMBEDDINGS_URL", os.environ.get("TEST_EMBEDDINGS_URL", "http://127.0.0.1:8001"))
    with TestClient(main.app) as client:
        cache = main.app.state.cache
        assert isinstance(cache.embeddings, RemoteEmbeddings)
        assert cache.embedding_dimension == 384
        assert np.isclose(np.linalg.norm(cache._embed("An English question")), 1.0)
        first = client.post("/v1/chat", json={"prompt": "What is the capital of France?"})
        assert first.status_code == 200
        assert first.json()["cached"] is False
        duplicate = client.post("/v1/chat", json={"prompt": "What is the capital of France?"})
        assert duplicate.json()["cached"] is True
        paraphrase = client.post("/v1/chat", json={
            "prompt": "Tell me the capital city of France.", "threshold": 0.8,
        })
        assert paraphrase.json()["cached"] is True
        different_context = client.post("/v1/chat", json={
            "prompt": "Tell me the capital city of France.", "system_prompt": "Different instructions",
        })
        assert different_context.json()["cached"] is False
        unrelated = client.post("/v1/chat", json={"prompt": "How do I reset my password?"})
        assert unrelated.json()["cached"] is False
        # A new cache instance uses the same model-specific index and persisted data.
        restarted = SemanticCache(redis_client, cache.embeddings, namespace=namespace)
        assert restarted.lookup("What is the capital of France?", "", main.app.state.llm.cache_model("test-model"), 0.0)[0] is not None


def test_revision_changes_select_a_different_index(redis_client, namespace, embeddings):
    old = SemanticCache(redis_client, embeddings, namespace=namespace)
    old.store("Question", "", "demo", 0.0, "old answer")
    embeddings.revision = "new-revision"
    new = SemanticCache(redis_client, embeddings, namespace=namespace)
    assert old.index_name != new.index_name
    assert new.lookup("Question", "", "demo", 0.0)[:2] == (None, None)
