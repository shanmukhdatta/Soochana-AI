"""Pydantic schemas for the Bujji AI financial assistant."""
from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str = Field(..., max_length=2000)


class AssistantChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, description="User's query")
    history: list[ChatMessage] = Field(default_factory=list, description="Recent conversation turns (up to 10)")
    mode: Literal["quick", "analyst"] = Field("quick", description="Current display mode preference")
    selected_signal_id: Optional[str] = Field(None, max_length=64, description="Optional hex ID of currently selected signal")


class AssistantChatResponse(BaseModel):
    reply: str
    status: Literal["ok", "unavailable", "rate_limited", "error"]
    mode: str
    disclaimer: str = "AI-generated explanations may contain errors. Research use only; not investment advice."
