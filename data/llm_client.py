"""OpenRouter LLM client for caption generation.

All enterprise LLM inference goes through OpenRouter. The API key is read
from the environment variable specified in config (default: OPENROUTER_API_KEY).
"""

import json
import os
import time

import requests
import yaml

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


def get_openrouter_key(config: dict | None = None) -> str:
    """Read OpenRouter API key from the configured env var."""
    env_var = "OPENROUTER_API_KEY"
    if config and "env" in config:
        env_var = config["env"].get("openrouter_token_var", env_var)
    key = os.environ.get(env_var)
    if not key:
        raise EnvironmentError(
            f"OpenRouter API key not found. Set the {env_var} environment variable."
        )
    return key


def chat_completion(
    messages: list[dict[str, str]],
    model: str = "anthropic/claude-sonnet-4",
    max_tokens: int = 300,
    temperature: float = 0.7,
    config: dict | None = None,
    max_retries: int = 3,
) -> str:
    """Call OpenRouter chat completion API.

    Args:
        messages: List of {"role": ..., "content": ...} dicts.
        model: OpenRouter model identifier.
        max_tokens: Max tokens in response.
        temperature: Sampling temperature.
        config: Project config dict (for reading env var names).
        max_retries: Number of retries on transient failures.

    Returns:
        The assistant's response text.
    """
    api_key = get_openrouter_key(config)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/sambt/hep-llava",
        "X-Title": "PhysLLaVA Caption Generation",
    }

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                wait = 2 ** attempt
                print(f"Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            raise
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"Connection error, retrying in {wait}s: {e}")
                time.sleep(wait)
                continue
            raise

    raise RuntimeError(f"Failed after {max_retries} retries")


def generate_caption_batch(
    prompts: list[str],
    system_prompt: str,
    model: str = "anthropic/claude-sonnet-4",
    max_tokens: int = 300,
    config: dict | None = None,
) -> list[str]:
    """Generate captions for a batch of prompts (sequential calls).

    Args:
        prompts: List of user prompts describing jet metadata.
        system_prompt: System prompt for the LLM.
        model: OpenRouter model ID.
        max_tokens: Max tokens per response.
        config: Project config dict.

    Returns:
        List of generated caption strings.
    """
    results = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        caption = chat_completion(
            messages, model=model, max_tokens=max_tokens, config=config
        )
        results.append(caption)
    return results
