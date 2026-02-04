from __future__ import annotations

import unittest

from fastapi import HTTPException

from mlx_openai_tts.auth import require_auth


class AuthTests(unittest.TestCase):
    def test_auth_disabled(self) -> None:
        require_auth(None, None)

    def test_missing_token(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            require_auth(None, "secret")
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertEqual(ctx.exception.detail, "Missing Bearer token")

    def test_invalid_token(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            require_auth("Bearer nope", "secret")
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertEqual(ctx.exception.detail, "Invalid token")

    def test_valid_token(self) -> None:
        require_auth("Bearer secret", "secret")


if __name__ == "__main__":
    unittest.main()
