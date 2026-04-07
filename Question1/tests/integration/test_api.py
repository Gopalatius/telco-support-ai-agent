from fastapi.testclient import TestClient

from telco_agent.api.dependencies import get_chat_service
from telco_agent.api.main import app
from telco_agent.domain.models import ChatRequest, ChatResponse


class StubChatService:
    def reply(self, request: ChatRequest) -> ChatResponse:
        assert request.message == "How much is the late fee?"
        return ChatResponse(
            reply="The late fee is IDR 50,000 after 14 days overdue.",
            escalate=False,
        )


def test_chat_endpoint_returns_expected_json() -> None:
    app.dependency_overrides[get_chat_service] = lambda: StubChatService()
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={
            "message": "How much is the late fee?",
            "history": [],
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "reply": "The late fee is IDR 50,000 after 14 days overdue.",
        "escalate": False,
    }

    app.dependency_overrides.clear()
