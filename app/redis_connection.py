"""Redis client configuration for local development and deployment."""

import os

from redis import Redis


def create_redis_client() -> Redis:
    """Create a reusable client; the first command opens the connection.

    REDIS_URL defaults to the service exposed by docker-compose.services.yml.
    Responses remain bytes so future vector embeddings can contain binary data.
    The caller owns the client and must close it when finished.
    """
    return Redis.from_url(
        os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
        decode_responses=False,
        socket_connect_timeout=5,
        socket_timeout=5,
        health_check_interval=30,
    )


if __name__ == "__main__":
    with create_redis_client() as client:
        client.ping()
        print("Redis: PONG")
