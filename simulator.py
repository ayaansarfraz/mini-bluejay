import asyncio
import json
import os
import sys
import time
    
import httpx
from anthropic import AsyncAnthropic
from dotenv import load_dotenv

from resilience import TokenBucket, retry

load_dotenv()

MODEL = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")
AGENT_URL = os.getenv("AGENT_URL", "http://localhost:8000/chat")
MAX_TURNS = 10
END_TAG = "[END]"

llm = AsyncAnthropic(max_retries=0)  # our own retry() handles failures


def persona_prompt(p: dict) -> str:
    return f"""You are role-playing a customer phoning Maple Dental. You are the CALLER, not the receptionist.

Name: {p["name"]}
Goal: {p["goal"]}
Mood: {p["mood"]}
Quirk: {p["quirk"]}

Rules:
- Stay in character. Speak like a real person on the phone: one or two short sentences, no stage directions.
- Pursue your goal. React naturally to what the receptionist says.
- When the call is over (you booked, you gave up, or you hung up), end your final line with the exact tag {END_TAG}.
"""


async def caller_turn(persona: dict, transcript: list[dict]) -> str:
    """Ask the persona LLM what the caller says next.

    The transcript is stored from the receptionist's point of view (caller = "user").
    The persona LLM sees the opposite: its own lines are "assistant", the receptionist's
    lines are "user".
    """
    if transcript:
        flipped = [
            {"role": "assistant" if m["role"] == "user" else "user", "content": m["content"]}
            for m in transcript
        ]
    else:
        # The API needs a first "user" message to start the conversation.
        flipped = [{"role": "user", "content": "(The phone rings. The receptionist picks up.)"}]

    resp = await llm.messages.create(
        model=MODEL,
        max_tokens=150,
        system=persona_prompt(persona),
        messages=flipped,
    )
    return resp.content[0].text.strip()


async def receptionist_turn(http: httpx.AsyncClient, transcript: list[dict]) -> str:
    # Send only role + content; the "ts" timestamps are for our own storage.
    messages = [{"role": m["role"], "content": m["content"]} for m in transcript]
    r = await http.post(AGENT_URL, json={"messages": messages}, timeout=60)
    r.raise_for_status()
    return r.json()["reply"]


async def limited(limiter: TokenBucket | None, fn, *args):
    """Take a rate-limit token, then call fn. Used per attempt, so retries are limited too."""
    if limiter:
        await limiter.acquire()
    return await fn(*args)


async def run_conversation(
    persona: dict, http: httpx.AsyncClient, limiter: TokenBucket | None = None
) -> dict:
    """Run one full call: persona talks to the receptionist until done or MAX_TURNS.

    Each turn is retried on its own, so a hiccup on turn 8 doesn't redo turns 1-7.
    """
    transcript: list[dict] = []  # caller = "user", receptionist = "assistant"
    ended = False

    for _ in range(MAX_TURNS):
        caller_text = await retry(limited, limiter, caller_turn, persona, transcript)
        ended = END_TAG in caller_text
        caller_text = caller_text.replace(END_TAG, "").strip()
        if caller_text:
            transcript.append({"role": "user", "content": caller_text, "ts": time.time()})
        if ended or not caller_text:
            break

        reply = await retry(limited, limiter, receptionist_turn, http, transcript)
        transcript.append({"role": "assistant", "content": reply, "ts": time.time()})

    return {
        "persona_id": persona["id"],
        "persona_name": persona["name"],
        "ended_by_caller": ended,
        "turns": len(transcript),
        "transcript": transcript,
    }


async def main() -> None:
    with open("personas.json") as f:
        personas = {p["id"]: p for p in json.load(f)}
    persona = personas[sys.argv[1] if len(sys.argv) > 1 else "p1"]

    async with httpx.AsyncClient() as http:
        result = await run_conversation(persona, http)

    print(f"--- {result['persona_name']} ({persona['mood']}): {persona['goal']}")
    for m in result["transcript"]:
        who = "CALLER      " if m["role"] == "user" else "RECEPTIONIST"
        print(f"{who}: {m['content']}")
    print(f"--- turns={result['turns']} ended_by_caller={result['ended_by_caller']}")


if __name__ == "__main__":
    asyncio.run(main())
