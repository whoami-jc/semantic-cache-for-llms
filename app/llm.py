"""OpenAI Responses API adapter. Only successful, complete text is cacheable."""

import httpx

from app.settings import Settings


class LLMError(RuntimeError):
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


class OpenAIProvider:
    def __init__(self, settings: Settings, *, transport=None):
        self.default_model = settings.openai_model
        self.max_output_tokens = settings.openai_max_output_tokens
        key = settings.openai_api_key.get_secret_value().strip()
        self.configured = bool(key)
        self.client = httpx.Client(
            base_url="https://api.openai.com/v1/",
            headers={"Authorization": f"Bearer {key}"} if key else {},
            timeout=httpx.Timeout(settings.openai_timeout_seconds, connect=10),
            transport=transport,
        )

    def cache_model(self, model: str) -> str:
        # Separate real responses from demo data and different generation limits.
        return f"openai-responses-v1:{model}:max-output-tokens={self.max_output_tokens}"

    def generate(self, prompt: str, system_prompt: str, model: str, temperature: float) -> str:
        if not self.configured:
            raise LLMError("Set OPENAI_API_KEY in .env and restart the API", 503)
        payload = {
            "model": model, "input": prompt, "temperature": temperature,
            "max_output_tokens": self.max_output_tokens, "store": False,
        }
        if system_prompt:
            payload["instructions"] = system_prompt
        try:
            response = self.client.post("responses", json=payload)
        except httpx.TimeoutException as exc:
            raise LLMError("OpenAI request timed out", 504) from exc
        except httpx.HTTPError as exc:
            raise LLMError("OpenAI is unavailable", 502) from exc
        if response.status_code in (401, 403):
            raise LLMError("OpenAI authentication or model access failed; check server configuration", 503)
        if response.status_code == 429:
            raise LLMError("OpenAI rate limit or quota exceeded", 503)
        if not response.is_success:
            # Do not expose upstream bodies, request headers, or credentials.
            raise LLMError("OpenAI rejected the request; check model and generation parameters", 502)
        try:
            data = response.json()
            if data.get("status") != "completed" or data.get("error") is not None:
                raise ValueError("Incomplete response")
            texts = []
            for item in data["output"]:
                if item.get("type") != "message":
                    continue
                if item.get("status") != "completed":
                    raise ValueError("Incomplete message")
                for part in item["content"]:
                    if part.get("type") == "refusal":
                        raise ValueError("Refusal is not cacheable")
                    if part.get("type") == "output_text":
                        texts.append(part["text"])
            text = "".join(texts)
            if not text.strip():
                raise ValueError("No output text")
            return text
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise LLMError("OpenAI returned no complete usable text response", 502) from exc

    def close(self):
        self.client.close()


def create_llm(settings: Settings) -> OpenAIProvider:
    return OpenAIProvider(settings)
