import json

import httpx
import pytest

from app.llm import OpenAIProvider, LLMError
from app.settings import Settings


def provider(handler, key="test-key"):
    return OpenAIProvider(Settings(_env_file=None, openai_api_key=key), transport=httpx.MockTransport(handler))


def completed(text="A real-shaped answer"):
    return {"status": "completed", "output": [
        {"type": "reasoning", "summary": []},
        {"type": "message", "status": "completed", "content": [{"type": "output_text", "text": text}]},
    ]}


def test_responses_request_and_text_extraction():
    def handler(request):
        assert str(request.url) == "https://api.openai.com/v1/responses"
        assert request.headers["Authorization"] == "Bearer test-key"
        assert json.loads(request.content) == {
            "model": "test-model", "input": "Question", "instructions": "Answer in English",
            "temperature": 0.2, "max_output_tokens": 1024, "store": False,
        }
        return httpx.Response(200, json=completed())
    client = provider(handler)
    try:
        assert client.generate("Question", "Answer in English", "test-model", 0.2) == "A real-shaped answer"
    finally:
        client.close()
    assert client.client.is_closed


@pytest.mark.parametrize("upstream,status", [(401,503), (403,503), (429,503), (400,502), (500,502)])
def test_http_failures_hide_upstream_body(upstream, status):
    client = provider(lambda request: httpx.Response(upstream, text="sensitive upstream detail"))
    try:
        with pytest.raises(LLMError) as error:
            client.generate("Question", "", "test-model", 0)
        assert error.value.status_code == status
        assert "sensitive" not in str(error.value)
    finally:
        client.close()


@pytest.mark.parametrize("body", [
    {"status": "incomplete", "output": []}, completed(""), {},
    {"status": "completed", "output": [{"type": "message", "status": "completed",
     "content": [{"type": "refusal", "refusal": "Cannot answer"}]}]},
])
def test_unusable_responses_are_not_returned(body):
    client = provider(lambda request: httpx.Response(200, json=body))
    try:
        with pytest.raises(LLMError):
            client.generate("Question", "", "test-model", 0)
    finally:
        client.close()


@pytest.mark.parametrize("exception,status", [(httpx.ReadTimeout,504), (httpx.ConnectError,502)])
def test_transport_failures(exception, status):
    def handler(request):
        raise exception("offline", request=request)
    client = provider(handler)
    try:
        with pytest.raises(LLMError) as error:
            client.generate("Question", "", "test-model", 0)
        assert error.value.status_code == status
    finally:
        client.close()


def test_missing_key_never_sends_a_request():
    def forbidden(request):
        pytest.fail("No request should be sent without a key")
    client = provider(forbidden, key="")
    try:
        with pytest.raises(LLMError, match="OPENAI_API_KEY"):
            client.generate("Question", "", "test-model", 0)
    finally:
        client.close()


def test_dotenv_settings_and_environment_precedence(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    path = tmp_path / ".env"
    path.write_text("OPENAI_API_KEY=test-secret\nOPENAI_MODEL=file-model\n")
    monkeypatch.setenv("OPENAI_MODEL", "environment-model")
    config = Settings(_env_file=path)
    assert config.openai_api_key.get_secret_value() == "test-secret"
    assert config.openai_model == "environment-model"
    assert "test-secret" not in repr(config)
