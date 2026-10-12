"""Local API adaptation; keep upstream reward-code extraction unchanged."""

import os
import re
import time

import openai


class QminiLLMManager:
    def __init__(self, config, num_suggestions, system_prompt):
        # Do not call the upstream constructor: it auto-selects Azure from the
        # process environment, whereas profiles explicitly select their provider.
        key_name = config["api_key_env"]
        key = os.environ.get(key_name)
        if not key:
            raise ValueError(f"Missing API key environment variable: {key_name}")
        if num_suggestions < 1:
            raise ValueError("num_suggestions must be positive")
        self._gpt_model = config["model"]
        self._num_suggestions = num_suggestions
        self._temperature = config.get("temperature", 1.0)
        self._supports_n = config.get("supports_n", True)
        self._stream = config.get("stream", False)
        self._prompts = [{"role": "system", "content": system_prompt}]
        if config["provider"] == "azure":
            self._client = openai.AzureOpenAI(
                api_key=key, azure_endpoint=config["base_url"], api_version=config["api_version"],
                timeout=config.get("timeout_seconds", 300), max_retries=config.get("max_retries", 0),
            )
        else:
            self._client = openai.OpenAI(
                api_key=key, base_url=config.get("base_url") or "https://api.openai.com/v1",
                timeout=config.get("timeout_seconds", 300), max_retries=config.get("max_retries", 0),
            )

    @staticmethod
    def extract_code_from_response(response: str) -> str:
        """Match the upstream Eureka convention: use the last Python code block."""
        matches = re.findall(r"```python(.*?)```", response, re.DOTALL)
        return matches[-1].lstrip("\n") if matches else ""

    def prompt(self, user_prompt, assistant_prompt=None):
        # Commit history only after a successful request, so retries do not add
        # duplicate turns. Retain the original task and latest feedback pair.
        messages = list(self._prompts)
        if assistant_prompt is not None:
            messages.append({"role": "assistant", "content": assistant_prompt})
        messages.append({"role": "user", "content": user_prompt})
        if len(messages) > 4:
            messages = messages[:2] + messages[-2:]
        kwargs = {"model": self._gpt_model, "messages": messages}
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        if self._supports_n:
            kwargs["n"] = self._num_suggestions
        if self._stream:
            kwargs["stream"] = True
        raw_outputs = []
        request_count = 1 if self._supports_n else self._num_suggestions
        for request_idx in range(request_count):
            started = time.monotonic()
            print(
                f"[LLM] Requesting {self._gpt_model} "
                f"({request_idx + 1}/{request_count}, {len(messages)} messages)...",
                flush=True,
            )
            try:
                response = self._client.chat.completions.create(**kwargs)
                expected = self._num_suggestions if self._supports_n else 1
                if self._stream:
                    print("[LLM] Stream opened; receiving tokens...", flush=True)
                    response_texts = [""] * expected
                    last_progress = time.monotonic()
                    try:
                        for chunk in response:
                            for choice in chunk.choices:
                                index = choice.index
                                if index < 0 or index >= expected:
                                    raise ValueError(f"API returned unexpected streamed choice index {index}")
                                content = getattr(choice.delta, "content", None)
                                if content:
                                    response_texts[index] += content
                            now = time.monotonic()
                            if now - last_progress >= 15:
                                print(f"[LLM] Still receiving ({now - started:.0f}s)...", flush=True)
                                last_progress = now
                    finally:
                        close = getattr(response, "close", None)
                        if close is not None:
                            close()
                else:
                    if len(response.choices) != expected:
                        raise ValueError(f"API returned {len(response.choices)} choices; expected {expected}")
                    response_texts = [choice.message.content for choice in response.choices]
            except Exception as exc:
                elapsed = time.monotonic() - started
                raise RuntimeError(
                    f"LLM request to {self._gpt_model} failed after {elapsed:.1f}s: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            print(f"[LLM] Response received in {time.monotonic() - started:.1f}s.", flush=True)
            for content in response_texts:
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("API returned no text in choices[].message.content")
                raw_outputs.append(content)
        reward_strings = [self.extract_code_from_response(text) for text in raw_outputs]
        self._prompts = messages
        return {"reward_strings": reward_strings, "raw_outputs": raw_outputs}
