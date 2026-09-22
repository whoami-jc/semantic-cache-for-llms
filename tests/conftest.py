import os
from uuid import uuid4

import pytest
from redis import Redis


@pytest.fixture
def namespace():
    return f"test-semantic-cache:{uuid4().hex}"


@pytest.fixture
def redis_client(namespace):
    url = os.environ.get("TEST_REDIS_URL", "redis://localhost:6379/0")
    client = Redis.from_url(url, socket_connect_timeout=5, socket_timeout=5)
    client.ping()
    try:
        yield client
    finally:
        for index in client.execute_command("FT._LIST"):
            if index.decode().startswith(namespace + ":"):
                client.ft(index.decode()).dropindex(delete_documents=True)
        for key in client.scan_iter(match=namespace + ":*"):
            client.delete(key)
        client.close()


class StubEmbeddings:
    """Deterministic vectors for infrastructure tests, never used by the API."""
    model_name = "test-embeddings"
    revision = "v1"
    dimension = 512

    def encode(self, text):
        import numpy as np
        return np.pad([1.0], (0, self.dimension - 1)).astype(np.float32)

    def close(self):
        pass


@pytest.fixture
def embeddings():
    return StubEmbeddings()


@pytest.fixture
def llm(monkeypatch):
    """Exercise the real adapter over a mock transport; never call paid APIs."""
    import httpx
    from app import main
    from app.llm import OpenAIProvider
    from app.settings import Settings

    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={
            "status": "completed", "output": [{
                "type": "message", "status": "completed",
                "content": [{"type": "output_text", "text": "A generated test answer."}],
            }],
        })

    providers = []

    def factory(settings):
        provider = OpenAIProvider(
            Settings(_env_file=None, openai_api_key="test-key", openai_model="test-model"),
            transport=httpx.MockTransport(handler),
        )
        provider.calls = calls
        providers.append(provider)
        return provider

    monkeypatch.setattr(main, "create_llm", factory)
    yield calls
    for provider in providers:
        provider.close()
