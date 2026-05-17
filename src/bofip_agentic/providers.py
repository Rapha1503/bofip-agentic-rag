"""LLM provider configuration — single source of truth for all scripts."""
from __future__ import annotations

PROVIDERS: dict[str, dict] = {
    "DeepSeek": {
        "base_url": "https://api.deepseek.com/v1",
        "models": ["deepseek-v4-flash", "deepseek-v4-pro"],
        "default_model": "deepseek-v4-flash",
        "env_key": "DEEPSEEK_API_KEY",
    },
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-5.5", "gpt-5.4-mini", "gpt-5-mini", "gpt-4.1"],
        "default_model": "gpt-5.4-mini",
        "env_key": "OPENAI_API_KEY",
    },
    "Anthropic": {
        "base_url": "https://api.anthropic.com/v1",
        "models": ["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-7"],
        "default_model": "claude-haiku-4-5",
        "env_key": "ANTHROPIC_API_KEY",
    },
    "Mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "models": ["mistral-small-4", "mistral-large-3", "mistral-medium-3.5"],
        "default_model": "mistral-small-4",
        "env_key": "MISTRAL_API_KEY",
    },
    "Google": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "models": ["gemini-3.1-flash-lite", "gemini-3.1-flash", "gemini-3.1-pro"],
        "default_model": "gemini-3.1-flash",
        "env_key": "GEMINI_API_KEY",
    },
    "Groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "models": ["llama-4-maverick", "meta-llama/llama-4-scout"],
        "default_model": "llama-4-maverick",
        "env_key": "GROQ_API_KEY",
    },
    "Together": {
        "base_url": "https://api.together.xyz/v1",
        "models": ["meta-llama/Llama-4-Maverick", "deepseek-ai/DeepSeek-V3"],
        "default_model": "meta-llama/Llama-4-Maverick",
        "env_key": "TOGETHER_API_KEY",
    },
}


def resolve_provider(name: str) -> dict | None:
    """Case-insensitive provider lookup."""
    for key in PROVIDERS:
        if key.lower() == name.lower():
            return PROVIDERS[key]
    return None
