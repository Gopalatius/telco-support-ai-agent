from fastapi.testclient import TestClient

from telco_agent.api.main import app
from telco_agent.api.routes import health as health_module


def test_health_endpoint_returns_ok(monkeypatch) -> None:
    class StubStore:
        def __init__(self) -> None:
            self.called = False

        def healthcheck(self) -> bool:
            self.called = True
            return True

    store = StubStore()
    monkeypatch.setattr(health_module, "get_settings", lambda: object())
    monkeypatch.setattr(health_module, "get_vector_store", lambda: store)

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert store.called is True


def test_health_endpoint_returns_503_when_dependency_is_unhealthy(monkeypatch) -> None:
    class StubStore:
        def healthcheck(self) -> bool:
            return False

    monkeypatch.setattr(health_module, "get_settings", lambda: object())
    monkeypatch.setattr(health_module, "get_vector_store", lambda: StubStore())

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 503
    assert response.json() == {"detail": "dependency healthcheck failed"}


def test_health_endpoint_returns_503_when_dependency_raises(monkeypatch) -> None:
    class StubStore:
        def healthcheck(self) -> bool:
            raise RuntimeError("qdrant unavailable")

    monkeypatch.setattr(health_module, "get_settings", lambda: object())
    monkeypatch.setattr(health_module, "get_vector_store", lambda: StubStore())

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 503
    assert response.json() == {"detail": "dependency healthcheck failed"}


def test_lifespan_configures_logging(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "telco_agent.api.main.configure_logging",
        lambda: calls.append("configured"),
    )

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert calls == ["configured"]
