import httpx
import numpy as np
import pytest

from app.embeddings import RemoteEmbeddings, EmbeddingServiceError


META = {"model": "test-model", "revision": "pinned-revision", "dimension": 2}


def make_provider(payload):
    def handler(request):
        return httpx.Response(200, json=META if request.url.path == "/health" else payload)
    return RemoteEmbeddings("http://embeddings", transport=httpx.MockTransport(handler))


def test_valid_normalized_embedding():
    provider = make_provider({**META, "embeddings": [[0.6, 0.8]]})
    try:
        assert np.allclose(provider.encode("An English question"), [0.6, 0.8])
    finally:
        provider.close()
    assert provider.client.is_closed


@pytest.mark.parametrize("changes", [
    {"revision": "different-revision"}, {"model": "different-model"},
    {"dimension": 3}, {"embeddings": [[1.0]]}, {"embeddings": [[0.0, 0.0]]},
    {"embeddings": [[1.0, 0.0], [1.0, 0.0]]}, {"embeddings": [["NaN", 0.0]]},
])
def test_rejects_incompatible_responses(changes):
    provider = make_provider({**META, "embeddings": [[1.0, 0.0]], **changes})
    try:
        with pytest.raises(EmbeddingServiceError):
            provider.encode("An English question")
    finally:
        provider.close()


def test_startup_failure_is_explicit():
    with pytest.raises(EmbeddingServiceError):
        RemoteEmbeddings("http://embeddings", transport=httpx.MockTransport(
            lambda request: httpx.Response(503)
        ))


def test_timeout_is_reported_without_fallback():
    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json=META)
        raise httpx.ReadTimeout("offline", request=request)
    provider = RemoteEmbeddings("http://embeddings", transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(EmbeddingServiceError):
            provider.encode("An English question")
    finally:
        provider.close()
