from hashlib import sha256
from time import perf_counter
import re
import json

from redis import Redis
from redis.exceptions import ResponseError
from redis.commands.search.field import TagField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

import numpy as np

from app.embeddings import EmbeddingProvider
from app.support_policy import SupportFAQPolicy


class SemanticCache:
    def __init__(
        self, redis_client: Redis, embeddings: EmbeddingProvider,
        threshold: float = 0.90, ttl_seconds: int = 3600,
        namespace: str = "semantic-cache",
        policy: SupportFAQPolicy | None = None,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if not re.fullmatch(r"[a-zA-Z0-9:_-]+", namespace):
            raise ValueError("namespace must contain only letters, numbers, :, _, or -")
        self.redis = redis_client
        self.ttl_seconds = ttl_seconds
        self.embeddings = embeddings
        self.embedding_dimension = embeddings.dimension
        self.threshold = threshold
        self.policy = policy
        identity = [embeddings.model_name, embeddings.revision, "normalized-v1"]
        if policy is not None:
            identity.append(policy.revision)
        backend = json.dumps(identity)
        self.prefix = f"{namespace}:v2:{sha256(backend.encode()).hexdigest()}:{self.embedding_dimension}:"
        self.index_name = self.prefix + "index"
        self._ensure_index()

    def _ensure_index(self) -> None:
        try:
            self.redis.ft(self.index_name).create_index(
                [
                    TagField("context"),
                    VectorField("embedding", "FLAT", {
                        "TYPE": "FLOAT32", "DIM": self.embedding_dimension,
                        "DISTANCE_METRIC": "COSINE",
                    }),
                ],
                definition=IndexDefinition(prefix=[self.prefix + "entry:"], index_type=IndexType.HASH),
            )
        except ResponseError as exc:
            # Concurrent API instances can attempt to create the same index.
            if "index already exists" not in str(exc).lower():
                raise

    def _vector(self, prompt: str) -> np.ndarray:
        vector = np.asarray(self._embed(prompt), dtype="<f4")
        if vector.shape != (self.embedding_dimension,) or not np.isfinite(vector).all():
            raise ValueError("Embedding must be a finite vector with the index dimension")
        return vector

    def _embed(self, text: str) -> np.ndarray:
        return self.embeddings.encode(text)

    def _context_id(self, system_prompt: str, model: str, temperature: float) -> str:
        # Structured encoding avoids collisions from separators inside user input.
        value = json.dumps([system_prompt, model, float(temperature) or 0.0])
        return sha256(value.encode()).hexdigest()

    def _entry_key(self, prompt: str, system_prompt: str, model: str, temperature: float) -> str:
        context = self._context_id(system_prompt, model, temperature)
        return f"{self.prefix}entry:{context}:{sha256(prompt.encode()).hexdigest()}"

    def _result(self, response, similarity, started):
        self.redis.hincrby(self.prefix + "stats", "hits" if response is not None else "misses", 1)
        return response, similarity, (perf_counter() - started) * 1000

    def lookup(
        self,
        prompt: str,
        system_prompt: str,
        model: str,
        temperature: float,
        threshold: float | None = None,
    ) -> tuple[str | None, float | None, float]:
        started = perf_counter()
        if self.policy is not None and not self.policy.allows(prompt):
            return self._result(None, None, started)
        exact = self.redis.hget(self._entry_key(prompt, system_prompt, model, temperature), "response")
        if exact is not None:
            return self._result(exact.decode("utf-8"), 1.0, started)

        vector = self._vector(prompt)
        # A zero vector has no defined cosine similarity; exact matching still works.
        if not np.any(vector):
            return self._result(None, None, started)
        context = self._context_id(system_prompt, model, temperature)
        query = (
            Query(f"(@context:{{{context}}})=>[KNN 1 @embedding $vector AS distance]")
            .sort_by("distance").return_fields("prompt", "response", "distance").paging(0, 1).dialect(2)
        )
        result = self.redis.ft(self.index_name).search(query, query_params={"vector": vector.tobytes()})
        best_response = None
        best_similarity = None
        if result.docs:
            doc = result.docs[0]
            if hasattr(doc, "response") and hasattr(doc, "distance"):
                best_response = doc.response
                best_similarity = max(-1.0, min(1.0, 1.0 - float(doc.distance)))

        cutoff = threshold if threshold is not None else self.threshold
        if best_similarity is not None and best_similarity >= cutoff:
            return self._result(best_response, best_similarity, started)
        return self._result(None, best_similarity, started)

    def store(self, prompt: str, system_prompt: str, model: str, temperature: float, response: str) -> None:
        if self.policy is not None and not self.policy.allows(prompt):
            return
        vector = self._vector(prompt)
        key = self._entry_key(prompt, system_prompt, model, temperature)
        mapping = {
            "prompt": prompt, "response": response,
            "context": self._context_id(system_prompt, model, temperature),
        }
        if np.any(vector):
            mapping["embedding"] = vector.tobytes()
        # Store the hash and its TTL atomically, including overwrites.
        with self.redis.pipeline(transaction=True) as pipe:
            pipe.delete(key)
            pipe.hset(key, mapping=mapping)
            pipe.expire(key, self.ttl_seconds)
            pipe.execute()

    def stats(self) -> dict[str, float | int]:
        hits, misses = self.redis.hmget(self.prefix + "stats", "hits", "misses")
        hits, misses = int(hits or 0), int(misses or 0)
        total = hits + misses
        keys = set(self.redis.scan_iter(match=self.prefix + "entry:*", count=100))
        entries = sum(self.redis.exists(key) for key in keys)
        return {
            "entries": entries,
            "hits": hits,
            "misses": misses,
            "hit_rate": hits / total if total else 0.0,
        }

    def clear(self) -> None:
        # Never flush the database: it may contain data from other applications.
        for key in self.redis.scan_iter(match=self.prefix + "*", count=100):
            self.redis.delete(key)
