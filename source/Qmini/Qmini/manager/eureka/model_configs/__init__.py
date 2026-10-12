"""Load explicit LLM profiles without importing Isaac Sim or API clients."""

import json
from pathlib import Path


def load_profile(name, model=None, temperature=None):
    """Accept a bundled profile name or a JSON path relative to the working directory."""
    path = Path(name).expanduser()
    if not path.is_file():
        path = Path(__file__).parent / f"{name}.json"
    with path.open(encoding="utf-8") as stream:
        cfg = json.load(stream)
    allowed = {
        "provider", "model", "base_url", "api_key_env", "api_version",
        "temperature", "supports_n", "timeout_seconds", "max_retries",
        "stream",
    }
    if not isinstance(cfg, dict) or set(cfg) - allowed:
        raise ValueError("Invalid model profile fields; store only the API key environment-variable name.")
    if model is not None:
        cfg["model"] = model
    if temperature is not None:
        cfg["temperature"] = temperature
    if cfg.get("provider") not in {"openai", "openai_compatible", "azure"}:
        raise ValueError("provider must be openai, openai_compatible or azure")
    for field in ("model", "api_key_env"):
        if not isinstance(cfg.get(field), str) or not cfg[field].strip():
            raise ValueError(f"Set {field} in {path} (model may also be supplied with --gpt_model)")
    if cfg["provider"] != "openai" and not cfg.get("base_url"):
        raise ValueError("This provider requires base_url")
    if cfg.get("base_url") is not None:
        from urllib.parse import urlsplit

        url = urlsplit(cfg["base_url"])
        if url.scheme not in {"http", "https"} or not url.netloc or url.username or url.password or url.query:
            raise ValueError("base_url must be an HTTP(S) URL without credentials or query parameters")
    if cfg["provider"] == "azure" and not cfg.get("api_version"):
        raise ValueError("Azure requires api_version")
    if type(cfg.get("supports_n", True)) is not bool:
        raise ValueError("supports_n must be true or false")
    if type(cfg.get("stream", False)) is not bool:
        raise ValueError("stream must be true or false")
    timeout = cfg.get("timeout_seconds", 300)
    if type(timeout) not in (int, float) or timeout <= 0:
        raise ValueError("timeout_seconds must be a positive number")
    retries = cfg.get("max_retries", 0)
    if type(retries) is not int or retries < 0:
        raise ValueError("max_retries must be a non-negative integer")
    value = cfg.get("temperature", 1.0)
    if value is not None and (type(value) not in (int, float) or not 0 <= value <= 2):
        raise ValueError("temperature must be null or a number between 0 and 2")
    return cfg
