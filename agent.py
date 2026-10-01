import os

from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()

MODEL = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")
client = AsyncAnthropic()  # reads ANTHROPIC_API_KEY from env

SYSTEM_PROMPT = """You're the receptionist at Maple Dental. Open slots are Tuesday 3pm, Wednesday 10am, Friday 2pm. Book the caller into one.

Keep replies to one or two short sentences, like a real phone call. Only offer these three slots. Don't invent prices, dentists, or policies. Once the caller picks a slot and gives their name, confirm the booking."""

app = FastAPI(title="Maple Dental receptionist")


class Message(BaseModel):
    role: str  # "user" = caller, "assistant" = receptionist
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


class ChatResponse(BaseModel):
    reply: str


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    # Stateless: the caller sends the whole conversation each turn.
    resp = await client.messages.create(
        model=MODEL,
        max_tokens=200,
        system=SYSTEM_PROMPT,
        messages=[m.model_dump() for m in req.messages],
    )
    return ChatResponse(reply=resp.content[0].text)
