"""Offline tests: python -m unittest discover -s source/Qmini/Qmini/manager/eureka -p 'test_model_config.py'."""

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).parent


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


profiles = load_file("eureka_test_profiles", ROOT / "model_configs" / "__init__.py")


class ModelConfigTests(unittest.TestCase):
    def test_deepseek_profile(self):
        cfg = profiles.load_profile("deepseek")
        self.assertEqual(cfg["provider"], "openai_compatible")
        self.assertEqual(cfg["model"], "deepseek-flash")
        self.assertEqual(cfg["base_url"], "https://api.deepseek.com")
        self.assertEqual(cfg["api_key_env"], "DEEPSEEK_API_KEY")
        self.assertIsNone(cfg["temperature"])
        self.assertFalse(cfg["supports_n"])
        self.assertTrue(cfg["stream"])
        self.assertEqual(cfg["timeout_seconds"], 300)
        self.assertEqual(cfg["max_retries"], 0)

    def test_templates_require_configuration(self):
        for name in ("openai", "compatible", "azure"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                profiles.load_profile(name)

    def test_overrides(self):
        cfg = profiles.load_profile("openai", model="test-model", temperature=0.5)
        self.assertEqual(cfg["model"], "test-model")
        self.assertEqual(cfg["temperature"], 0.5)
        with self.assertRaises(ValueError):
            profiles.load_profile("openai", model="test", temperature=float("nan"))

    def make_manager(self, supports_n=False, provider="openai_compatible", temperature=None):
        client = Mock()
        sdk = types.ModuleType("openai")
        sdk.OpenAI = Mock(return_value=client)
        sdk.AzureOpenAI = Mock(return_value=client)
        with patch.dict(sys.modules, {"openai": sdk}):
            module = load_file("eureka_test_llm", ROOT / "llm_manager.py")
        config = dict(provider=provider, model="test", base_url="https://example.invalid/v1",
                      api_key_env="EUREKA_TEST_KEY", api_version="test-version",
                      supports_n=supports_n, temperature=temperature)
        with patch.dict("os.environ", {"EUREKA_TEST_KEY": "test-only", "AZURE_OPENAI_API_KEY": "ignored"}):
            manager = module.QminiLLMManager(config, 2, "system")
        return manager, client, sdk

    @staticmethod
    def response(count):
        return types.SimpleNamespace(choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content="```python\ncode\n```")) for _ in range(count)])

    def test_separate_requests_omit_unsupported_parameters(self):
        manager, client, sdk = self.make_manager()
        client.chat.completions.create.return_value = self.response(1)
        result = manager.prompt("task")
        self.assertEqual(len(result["reward_strings"]), 2)
        self.assertEqual(client.chat.completions.create.call_count, 2)
        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertNotIn("n", kwargs)
        self.assertNotIn("temperature", kwargs)
        sdk.AzureOpenAI.assert_not_called()
        self.assertEqual(sdk.OpenAI.call_args.kwargs["base_url"], "https://example.invalid/v1")

    def test_batch_azure_and_history(self):
        manager, client, sdk = self.make_manager(True, "azure", 0.5)
        client.chat.completions.create.return_value = self.response(2)
        manager.prompt("task")
        manager.prompt("feedback1", "answer1")
        manager.prompt("feedback2", "answer2")
        self.assertEqual([m["content"] for m in manager._prompts], ["system", "task", "answer2", "feedback2"])
        self.assertEqual(client.chat.completions.create.call_args.kwargs["n"], 2)
        self.assertEqual(client.chat.completions.create.call_args.kwargs["temperature"], 0.5)
        sdk.AzureOpenAI.assert_called_once()
        sdk.OpenAI.assert_not_called()

    def test_streaming_response_is_joined(self):
        manager, client, _ = self.make_manager()
        manager._stream = True
        manager._num_suggestions = 1
        chunks = [
            types.SimpleNamespace(choices=[types.SimpleNamespace(
                index=0, delta=types.SimpleNamespace(content="```python\n"))]),
            types.SimpleNamespace(choices=[types.SimpleNamespace(
                index=0, delta=types.SimpleNamespace(content="code\n```"))]),
        ]
        stream = Mock()
        stream.__iter__ = Mock(return_value=iter(chunks))
        client.chat.completions.create.return_value = stream
        result = manager.prompt("task")
        self.assertEqual(result["reward_strings"], ["code\n"])
        self.assertTrue(client.chat.completions.create.call_args.kwargs["stream"])
        self.assertEqual(stream.close.call_count, 1)

    def test_invalid_response_keeps_history(self):
        manager, client, _ = self.make_manager(True)
        client.chat.completions.create.return_value = self.response(1)
        with self.assertRaises(RuntimeError):
            manager.prompt("task")
        self.assertEqual(len(manager._prompts), 1)
        response = self.response(2)
        response.choices[0].message.content = None
        client.chat.completions.create.return_value = response
        with self.assertRaises(ValueError):
            manager.prompt("task")


if __name__ == "__main__":
    unittest.main()
