"""One-shot smoke test: call OpenRouter Haiku 4.5, confirm JSON round-trip.
Run: source .env.local && python3 scripts/smoke_openrouter.py
"""
import json
import os
import re
import sys
from openai import OpenAI


def strip_code_fence(text: str) -> str:
    """Strip ```json ... ``` or ``` ... ``` wrappers that Anthropic via
    OpenRouter sometimes adds even when response_format=json_object is set."""
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```$", text, re.DOTALL)
    return m.group(1).strip() if m else text


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

resp = client.chat.completions.create(
    model="anthropic/claude-haiku-4.5",
    messages=[
        {"role": "system", "content": 'Reply with JSON only, no markdown fences: {"ok": true, "model": "<model name>"}'},
        {"role": "user", "content": "ping"},
    ],
    response_format={"type": "json_object"},
    temperature=0,
    timeout=30,
)
content = resp.choices[0].message.content
cleaned = strip_code_fence(content)
parsed = json.loads(cleaned)
print("Raw:", repr(content[:200]))
print("Parsed:", parsed)
assert parsed.get("ok") is True, f"unexpected response: {parsed}"
print("Smoke test PASSED")
