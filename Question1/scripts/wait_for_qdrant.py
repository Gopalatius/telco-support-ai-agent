import os
import time

import httpx


def main() -> None:
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333").rstrip("/")
    health_url = f"{qdrant_url}/healthz"
    deadline = time.monotonic() + 30

    while time.monotonic() < deadline:
        try:
            response = httpx.get(health_url, timeout=2.0)
            response.raise_for_status()
            print(f"Qdrant is ready at {health_url}")
            return
        except httpx.HTTPError:
            time.sleep(1)

    raise RuntimeError(f"Qdrant did not become ready within 30 seconds: {health_url}")


if __name__ == "__main__":
    main()
