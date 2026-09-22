import pytest

from app.cache import SemanticCache
from app.support_policy import SupportFAQPolicy
from app.validate_support import summarize


@pytest.mark.parametrize("prompt", [
    "Where is my refund?", "Cancel order A123", "Refund twenty euros", "Shipping to user@example.com",
    "What is the payment status now?", "Explain Python decorators", "Where is our delivery?",
])
def test_excluded_requests_do_not_embed_or_store(redis_client, namespace, embeddings, monkeypatch, prompt):
    cache = SemanticCache(redis_client, embeddings, namespace=namespace, policy=SupportFAQPolicy())

    def unexpected(text):
        pytest.fail("Excluded request must not call the model")

    monkeypatch.setattr(embeddings, "encode", unexpected)
    cache.store(prompt, "", "test", 0, "private answer")
    assert cache.lookup(prompt, "", "test", 0)[:2] == (None, None)
    assert cache.stats()["entries"] == 0


def test_scope_has_separate_index_and_keeps_faq_context_isolated(redis_client, namespace, embeddings):
    legacy = SemanticCache(redis_client, embeddings, namespace=namespace)
    scoped = SemanticCache(redis_client, embeddings, namespace=namespace, policy=SupportFAQPolicy())
    prompt = "What payment methods are accepted?"
    legacy.store(prompt, "store-a", "test", 0, "legacy")
    assert scoped.lookup(prompt, "store-a", "test", 0)[0] is None
    scoped.store(prompt, "store-a", "test", 0, "store-a answer")
    assert scoped.lookup(prompt, "store-a", "test", 0)[0] == "store-a answer"
    assert scoped.lookup(prompt, "store-b", "test", 0)[0] is None


def test_wrong_faq_is_an_error_even_when_query_is_cacheable():
    rows = [{"expected": "returns", "actual": "payments", "type": "paraphrase", "lookup_ms": 1}]
    result = summarize(rows)
    assert result["wrong_reuse"] == 1
    assert result["missed_positives"] == 1
    assert result["paraphrase_hits"] == 0
