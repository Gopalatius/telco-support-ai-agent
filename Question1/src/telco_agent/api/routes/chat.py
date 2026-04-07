from typing import Annotated

from fastapi import APIRouter, Depends

from telco_agent.api.dependencies import get_chat_service
from telco_agent.application.chat_service import ChatService
from telco_agent.domain.models import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    chat_service: Annotated[ChatService, Depends(get_chat_service)],
) -> ChatResponse:
    return chat_service.reply(request)
