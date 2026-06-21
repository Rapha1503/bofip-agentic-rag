"""LLM provider configuration used by the Streamlit app and eval scripts."""
from __future__ import annotations

PROVIDERS: dict[str, dict] = {
    "DeepSeek": {
        "base_url": "https://api.deepseek.com/v1",
        "models": ["deepseek-v4-flash", "deepseek-v4-pro", "deepseek-chat", "deepseek-reasoner"],
        "default_model": "deepseek-v4-flash",
        "env_key": "DEEPSEEK_API_KEY",
        "note": "DeepSeek v4 configure au 20/06/2026. deepseek-chat/reasoner restent des alias de compatibilite.",
    },
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-5.4-mini", "gpt-5.5", "gpt-5.4", "gpt-5.4-nano", "gpt-4.1"],
        "default_model": "gpt-5.4-mini",
        "env_key": "OPENAI_API_KEY",
        "note": "Modeles OpenAI configures au 20/06/2026. Mini par defaut pour limiter le cout.",
    },
    "Mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "models": ["mistral-small-latest", "mistral-medium-latest", "mistral-large-latest", "magistral-medium-latest"],
        "default_model": "mistral-small-latest",
        "env_key": "MISTRAL_API_KEY",
        "note": "Alias Mistral latest pour eviter de figer un modele retire.",
    },
    "Google": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "models": ["gemini-3.5-flash", "gemini-2.5-flash", "gemini-2.5-pro"],
        "default_model": "gemini-3.5-flash",
        "env_key": "GEMINI_API_KEY",
        "note": "Endpoint Gemini OpenAI-compatible configure pour BYOK.",
    },
}


def resolve_provider(name: str) -> dict | None:
    """Case-insensitive provider lookup."""
    for key, provider in PROVIDERS.items():
        if key.lower() == name.lower():
            return provider
    return None
