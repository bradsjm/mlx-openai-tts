from __future__ import annotations

import unittest

from mlx_openai_tts.registry import load_registry_payload


class RegistryTests(unittest.TestCase):
    def test_empty_models(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            load_registry_payload({"models": []})
        self.assertIn("at least one model", str(ctx.exception))

    def test_duplicate_model_id(self) -> None:
        payload = {
            "models": [
                {"id": "dup", "repo_id": "repo1"},
                {"id": "dup", "repo_id": "repo2"},
            ]
        }
        with self.assertRaises(RuntimeError) as ctx:
            load_registry_payload(payload)
        self.assertIn("Duplicate model id", str(ctx.exception))

    def test_default_model_missing(self) -> None:
        payload = {
            "default_model": "missing",
            "models": [{"id": "kokoro", "repo_id": "repo"}],
        }
        with self.assertRaises(RuntimeError) as ctx:
            load_registry_payload(payload)
        self.assertIn("default_model", str(ctx.exception))

    def test_default_voice_outside_list(self) -> None:
        payload = {
            "models": [
                {
                    "id": "kokoro",
                    "repo_id": "repo",
                    "voices": ["v1"],
                    "default_voice": "v2",
                }
            ]
        }
        with self.assertRaises(RuntimeError) as ctx:
            load_registry_payload(payload)
        self.assertIn("default_voice", str(ctx.exception))

    def test_default_voice_autofill(self) -> None:
        payload = {
            "models": [
                {
                    "id": "kokoro",
                    "repo_id": "repo",
                    "voices": ["v1", "v2"],
                }
            ]
        }
        registry = load_registry_payload(payload)
        self.assertEqual(registry.default_model, "kokoro")
        self.assertEqual(registry.models_by_id["kokoro"].default_voice, "v1")

    def test_invalid_model_type(self) -> None:
        payload = {"models": [{"id": "bad", "repo_id": "repo", "model_type": "oops"}]}
        with self.assertRaises(RuntimeError) as ctx:
            load_registry_payload(payload)
        self.assertIn("model_type", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
