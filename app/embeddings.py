"""HTTP boundary to the Docker-only embedding model (no local inference)."""

from typing import Protocol

import httpx
import numpy as np
from pydantic import BaseModel, Field, StrictInt


class EmbeddingServiceError(RuntimeError):
    """The service is unavailable or its response is incompatible."""


class EmbeddingProvider(Protocol):
    model_name: str
    revision: str
    dimension: int

    def encode(self, text: str) -> np.ndarray: ...


class Metadata(BaseModel):
    model: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    dimension: StrictInt = Field(gt=0)


class RemoteEmbeddings:
    def __init__(self, url: str, *, transport=None):
        self.client = httpx.Client(
            base_url=url.rstrip("/"), timeout=httpx.Timeout(60, connect=5), transport=transport,
        )
        try:
            self.metadata = Metadata.model_validate(self._request("GET", "/health"))
        except (EmbeddingServiceError, ValueError, TypeError) as exc:
            self.client.close()
            raise EmbeddingServiceError("Could not initialize embedding service") from exc
        self.model_name = self.metadata.model
        self.revision = self.metadata.revision
        self.dimension = self.metadata.dimension

    def _request(self, method, path, **kwargs):
        try:
            response = self.client.request(method, path, **kwargs)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise EmbeddingServiceError("Embedding service request failed") from exc

    def encode(self, text: str) -> np.ndarray:
        data = self._request("POST", "/embed", json={"texts": [text]})
        try:
            if Metadata.model_validate(data) != self.metadata:
                raise ValueError("Model changed; restart the API to select a new index")
            vectors = np.asarray(data["embeddings"], dtype=np.float32)
            if vectors.shape != (1, self.dimension) or not np.isfinite(vectors).all():
                raise ValueError("Invalid embedding shape or values")
            if not np.isclose(np.linalg.norm(vectors[0]), 1.0, atol=1e-4):
                raise ValueError("Embedding must be normalized")
            return vectors[0]
        except (ValueError, KeyError, TypeError) as exc:
            raise EmbeddingServiceError("Invalid embedding service response") from exc

    def close(self):
        self.client.close()
