import time

import pytest
from redis.exceptions import ConnectionError

from app import cache as cache_module


def test_instances_share_entries_and_counters(redis_client, namespace, embeddings):
    first = cache_module.SemanticCache(redis_client, embeddings, namespace=namespace)
    first.lookup("hello", "", "demo", 0.0)
    first.store("hello", "", "demo", 0.0, "answer")
    second = cache_module.SemanticCache(redis_client, embeddings, namespace=namespace)
    assert second.lookup("hello", "", "demo", 0.0)[:2] == ("answer", 1.0)
    assert first.stats() == {"entries": 1, "hits": 1, "misses": 1, "hit_rate": 0.5}


def test_expired_entries_are_not_reused(redis_client, namespace, embeddings):
    cache = cache_module.SemanticCache(redis_client, embeddings, namespace=namespace, ttl_seconds=1)
    cache.store("hello", "", "demo", 0.0, "answer")
    key = next(redis_client.scan_iter(match=namespace + ":*entry:*"))
    assert 0 < redis_client.pttl(key) <= 1000
    time.sleep(1.1)
    assert cache.lookup("hello", "", "demo", 0.0)[:2] == (None, None)
    assert cache.stats()["entries"] == 0
    assert cache.lookup("hello again", "", "demo", 0.0, threshold=0.0)[:2] == (None, None)


def test_clear_preserves_other_data_and_repeated_store_is_unique(redis_client, namespace, embeddings):
    cache = cache_module.SemanticCache(redis_client, embeddings, namespace=namespace)
    foreign_key = namespace + ":other-app"
    redis_client.set(foreign_key, "keep")
    cache.store("hello", "", "demo", 0.0, "old")
    cache.store("hello", "", "demo", 0.0, "new")
    assert cache.stats()["entries"] == 1
    assert cache.lookup("hello", "", "demo", 0.0)[0] == "new"
    cache.clear()
    assert cache.stats() == {"entries": 0, "hits": 0, "misses": 0, "hit_rate": 0.0}
    assert redis_client.get(foreign_key) == b"keep"
    # Clearing documents preserves a usable index.
    cache.store("hello again", "", "demo", 0.0, "after clear")
    assert cache.lookup("hello", "", "demo", 0.0, threshold=0.0)[0] == "after clear"


def test_backend_failure_propagates(redis_client, namespace, embeddings, monkeypatch):
    cache = cache_module.SemanticCache(redis_client, embeddings, namespace=namespace)

    def unavailable(*args, **kwargs):
        raise ConnectionError("offline")

    monkeypatch.setattr(redis_client, "hget", unavailable)
    with pytest.raises(ConnectionError):
        cache.lookup("hello", "", "demo", 0.0)
