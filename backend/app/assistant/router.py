"""FastAPI router for Bujji AI Financial Assistant."""
from fastapi import APIRouter
from .schemas import AssistantChatRequest, AssistantChatResponse
from .service import AssistantService

router = APIRouter(prefix="/assistant", tags=["Bujji Financial Assistant"])


@router.post("/chat", response_model=AssistantChatResponse)
async def chat(request: AssistantChatRequest) -> AssistantChatResponse:
    """Send a question to Bujji financial assistant, receiving mode-tailored answers."""
    return await AssistantService.chat(request)
