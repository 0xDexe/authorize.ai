"""
AuthorizeAI — LLM Client
==========================
Thin wrapper around LLM API calls. Supports Anthropic (Claude) and
OpenAI (GPT-4o-mini) with automatic JSON parsing and retry logic.
"""

import json
import os
import time
from typing import Any


def call_llm(
    system_prompt: str,
    user_prompt: str,
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    max_retries: int = 2,
) -> dict | str:
    """
    Call the configured LLM and return parsed JSON or raw text.

    Provider resolution order:
    1. Explicit `provider` param
    2. AUTHORIZEAI_LLM_PROVIDER env var
    3. Auto-detect based on available API keys
    """
    provider = _resolve_provider(provider)

    for attempt in range(max_retries + 1):
        try:
            if provider == "anthropic":
                raw = _call_anthropic(
                    system_prompt, user_prompt,
                    model=model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            elif provider == "openai":
                raw = _call_openai(
                    system_prompt, user_prompt,
                    model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            elif provider == "ollama":
                raw = _call_ollama(
                    system_prompt, user_prompt,
                    model=model or os.getenv("OLLAMA_MODEL", "llama3.2"),
                    temperature=temperature,
                )
            elif provider == "cloudflare":
                raw = _call_cloudflare(
                    system_prompt, user_prompt,
                    model=model or os.getenv("CLOUDFLARE_MODEL", "@cf/meta/llama-3.1-8b-instruct"),
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")

            # Try to parse as JSON
            return _parse_json_response(raw)

        except Exception as e:
            if attempt < max_retries:
                time.sleep(1 * (attempt + 1))
                continue
            raise RuntimeError(
                f"LLM call failed after {max_retries + 1} attempts: {e}"
            ) from e


def _resolve_provider(explicit: str | None) -> str:
    if explicit:
        return explicit.lower()
    env = os.getenv("AUTHORIZEAI_LLM_PROVIDER", "").lower()
    if env:
        return env
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("CLOUDFLARE_API_TOKEN"):
        return "cloudflare"
    # Fall back to Ollama (local or remote via OLLAMA_HOST)
    return "ollama"


def _call_anthropic(
    system: str, user: str, model: str,
    temperature: float, max_tokens: int,
) -> str:
    try:
        import anthropic
    except ImportError:
        raise ImportError("pip install anthropic")

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text


def _call_openai(
    system: str, user: str, model: str,
    temperature: float, max_tokens: int,
) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai")

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content


def _call_ollama(
    system: str, user: str, model: str, temperature: float,
) -> str:
    try:
        import ollama
    except ImportError:
        raise ImportError(
            "pip install ollama  (also install Ollama from https://ollama.com)"
        )

    # OLLAMA_HOST allows pointing to a remote Ollama server
    # e.g. OLLAMA_HOST=http://192.168.1.100:11434
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    client = ollama.Client(host=host)

    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options={"temperature": temperature},
    )
    return response["message"]["content"]


def _call_cloudflare(
    system: str, user: str, model: str,
    temperature: float, max_tokens: int,
) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai")

    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    api_token = os.getenv("CLOUDFLARE_API_TOKEN")

    if not account_id or not api_token:
        raise EnvironmentError(
            "Set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN in .env"
        )

    client = OpenAI(
        api_key=api_token,
        base_url=f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
    )
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content


def _parse_json_response(raw: str) -> dict | str:
    """
    Attempt to parse LLM output as JSON.
    Handles common issues: markdown fences, leading text, etc.
    Returns dict if successful, raw string otherwise.
    """
    text = raw.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    # Return raw text as fallback
    return text
