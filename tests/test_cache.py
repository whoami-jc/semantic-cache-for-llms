import numpy as np
import pytest

from app import cache as cache_module


@pytest.fixture
def cache(monkeypatch, redis_client, namespace, embeddings):
    # Deterministic vectors keep isolation tests independent of model downloads.
    instance = cache_module.SemanticCache(redis_client, embeddings, namespace=namespace)
    monkeypatch.setattr(instance, "_embed", lambda text: np.pad([1.0, 0.0], (0, 510)))
    return instance


@pytest.mark.parametrize("prompt", ["original question", "rephrased question"])
@pytest.mark.parametrize(
    "changed_context",
    [
        {"system_prompt": "different instructions"},
        {"model": "different-model"},
        {"temperature": 1.0},
    ],
)
def test_different_context_misses_even_with_identical_embeddings(cache, prompt, changed_context):
    context = {"system_prompt": "instructions", "model": "model", "temperature": 0.0}
    cache.store("original question", **context, response="private answer")

    response, similarity, _ = cache.lookup(prompt, **(context | changed_context), threshold=0.0)

    assert response is None
    assert similarity is None
    assert cache.stats() == {"entries": 1, "hits": 0, "misses": 1, "hit_rate": 0.0}


@pytest.mark.parametrize("prompt", ["original question", "rephrased question"])
def test_same_context_hits_when_other_context_was_stored_first(cache, prompt):
    cache.store("original question", "other instructions", "model", 0.0, "wrong answer")
    cache.store("original question", "instructions", "model", 0.0, "correct answer")

    response, similarity, _ = cache.lookup(prompt, "instructions", "model", 0.0)

    assert response == "correct answer"
    assert similarity == pytest.approx(1.0)
    assert cache.stats()["hits"] == 1
    assert cache.stats()["misses"] == 0


def test_closer_entry_in_other_context_does_not_hide_valid_match(cache, monkeypatch):
    vectors = {
        "unrelated context": np.pad([1.0, 0.0], (0, 510)),
        "valid question": np.pad([0.96, 0.28], (0, 510)),
        "new question": np.pad([1.0, 0.0], (0, 510)),
    }
    monkeypatch.setattr(cache, "_embed", vectors.__getitem__)
    cache.store("unrelated context", "other instructions", "model", 0.0, "wrong answer")
    cache.store("valid question", "instructions", "model", 0.0, "correct answer")

    response, similarity, _ = cache.lookup("new question", "instructions", "model", 0.0, threshold=0.9)

    assert response == "correct answer"
    assert similarity == pytest.approx(0.96)


def test_threshold_still_applies_within_context(cache, monkeypatch):
    cache.store("original question", "instructions", "model", 0.0, "answer")
    monkeypatch.setattr(cache, "_embed", lambda text: np.pad([0.8, 0.6], (0, 510)))

    response, similarity, _ = cache.lookup("new question", "instructions", "model", 0.0, threshold=0.9)

    assert response is None
    assert similarity == pytest.approx(0.8)
    assert cache.stats()["misses"] == 1


def test_semantic_lookup_uses_index_without_scan(cache, monkeypatch):
    cache.store("original question", "instructions", "model", 0.0, "A café serves coffee")

    def forbidden(*args, **kwargs):
        raise AssertionError("Lookup must use the vector index, not scan Redis keys")

    with monkeypatch.context() as patch:
        patch.setattr(cache.redis, "scan_iter", forbidden)
        assert cache.lookup("paraphrase", "instructions", "model", 0.0)[0] == "A café serves coffee"


def test_zero_vectors_only_allow_exact_hits(cache, monkeypatch):
    monkeypatch.setattr(cache, "_embed", lambda text: np.zeros(512))
    cache.store("!!!", "", "model", 0.0, "answer")
    assert cache.lookup("!!!", "", "model", 0.0)[0] == "answer"
    assert cache.lookup("???", "", "model", 0.0, threshold=0.0)[:2] == (None, None)


def test_invalid_dimension_is_rejected(cache, monkeypatch):
    monkeypatch.setattr(cache, "_embed", lambda text: np.ones(2))
    with pytest.raises(ValueError, match="dimension"):
        cache.store("question", "", "model", 0.0, "answer")
