"""Characterization + behavior tests for post-merge gemini-web2api.

Covers the three behaviors the upstream sync touched and that the live
deployment depends on:
1. Auth gate (upstream 9caacd0 / b99e340): rich auth — Bearer, x-api-key,
   x-goog-api-key, ?key= query param; 401 on invalid/missing key.
2. Proxy pool (local feature): rotation, failure cooldown, success reset,
   /status endpoint.
3. Config merge: upstream gemini_bl/default_model + local proxies keys.
"""

import json
import os
import sys
import threading
import time
import unittest
from http.client import HTTPConnection

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gemini_web2api.config import CONFIG, DEFAULT_CONFIG
from gemini_web2api.proxy_pool import POOL
from gemini_web2api.server import ThreadedServer, GeminiHandler

TEST_KEY = "sk-test-merged-key"
TEST_PORT = 18123


def _auth_headers(key=None):
    return {"Authorization": f"Bearer {key}"} if key else {}


class ServerAuthTest(unittest.TestCase):
    """S1: auth gate behavior on the merged server."""

    @classmethod
    def setUpClass(cls):
        CONFIG["api_keys"] = [TEST_KEY]
        CONFIG["port"] = TEST_PORT
        CONFIG["proxies"] = []
        cls.server = ThreadedServer(("127.0.0.1", TEST_PORT), GeminiHandler)
        cls.t = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.t.start()
        time.sleep(0.3)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def _get(self, path, headers=None):
        conn = HTTPConnection("127.0.0.1", TEST_PORT, timeout=5)
        conn.request("GET", path, headers=headers or {})
        resp = conn.getresponse()
        body = resp.read()
        conn.close()
        return resp.status, body

    def test_bearer_key_authorized(self):
        status, _ = self._get("/v1/models", _auth_headers(TEST_KEY))
        self.assertEqual(status, 200, "valid Bearer key must pass")

    def test_missing_key_rejected(self):
        status, _ = self._get("/v1/models")
        self.assertEqual(status, 401, "missing key must 401")

    def test_wrong_key_rejected(self):
        status, _ = self._get("/v1/models", _auth_headers("sk-wrong"))
        self.assertEqual(status, 401, "wrong key must 401")

    def test_x_api_key_header_authorized(self):
        status, _ = self._get("/v1/models", {"x-api-key": TEST_KEY})
        self.assertEqual(status, 200, "x-api-key header must pass")

    def test_x_goog_api_key_header_authorized(self):
        status, _ = self._get("/v1/models", {"x-goog-api-key": TEST_KEY})
        self.assertEqual(status, 200, "x-goog-api-key header must pass")

    def test_query_key_authorized(self):
        status, _ = self._get(f"/v1/models?key={TEST_KEY}")
        self.assertEqual(status, 200, "?key= query param must pass")

    def test_query_wrong_key_rejected(self):
        status, _ = self._get("/v1/models?key=sk-wrong")
        self.assertEqual(status, 401, "?key= with wrong key must 401")

    def test_models_returns_model_list(self):
        status, body = self._get("/v1/models", _auth_headers(TEST_KEY))
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data["object"], "list")
        ids = [m["id"] for m in data["data"]]
        self.assertIn("gemini-3.6-flash", ids, "upstream model must be listed")

    def test_status_endpoint(self):
        status, body = self._get("/status")
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data["status"], "ok")
        self.assertIn("proxy_pool", data, "local /status must expose proxy_pool")


class ProxyPoolTest(unittest.TestCase):
    """S2: proxy pool rotation/cooldown/success-reset behavior."""

    def setUp(self):
        CONFIG["proxies"] = []
        CONFIG["proxy_rotate"] = {"enabled": True, "cooldown_sec": 5, "fail_threshold": 1}
        POOL.configure_from_config()
        CONFIG["proxies"] = ["socks5://exit-a:1080", "socks5://exit-b:1080"]
        POOL.configure_from_config()

    def test_configure_loads_all_exits(self):
        self.assertEqual(len(POOL.all_proxies()), 2)

    def test_current_is_first_exit(self):
        self.assertEqual(POOL.current(), "socks5://exit-a:1080")

    def test_failure_rotates_current(self):
        POOL.mark_failure("socks5://exit-a:1080", "conn refused")
        self.assertEqual(POOL.current(), "socks5://exit-b:1080", "fail must rotate to next exit")

    def test_failure_puts_exit_in_cooldown(self):
        POOL.mark_failure("socks5://exit-a:1080", "conn refused")
        candidates = POOL.candidates()
        self.assertNotIn("socks5://exit-a:1080", candidates, "failed exit must be cooled down")

    def test_success_resets_failure_state(self):
        POOL.mark_failure("socks5://exit-a:1080", "conn refused")
        POOL.mark_success("socks5://exit-a:1080")
        state = POOL.status()["exits"][0]
        self.assertEqual(state["proxy"], "socks5://exit-a:1080")
        self.assertEqual(state["consecutive_fails"], 0, "success must reset fail counter")


class ConfigMergeTest(unittest.TestCase):
    """S3: merged DEFAULT_CONFIG carries upstream values + local proxy keys."""

    def test_upstream_gemini_bl(self):
        self.assertEqual(DEFAULT_CONFIG["gemini_bl"], "boq_assistant-bard-web-server_20260716.08_p0")

    def test_upstream_default_model(self):
        self.assertEqual(DEFAULT_CONFIG["default_model"], "gemini-3.6-flash")

    def test_local_proxy_keys_present(self):
        self.assertIn("proxies", DEFAULT_CONFIG)
        self.assertIn("proxy_rotate", DEFAULT_CONFIG)
        self.assertEqual(DEFAULT_CONFIG["proxy_rotate"]["cooldown_sec"], 300)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class BardErrorDetectionTest(unittest.TestCase):
    """S4: JSON-form BardErrorInfo must be detected (upstream regex misses it)."""

    def _json_error_payload(self):
        return (
            ")]}'\n\n121\n"
            '[["wrb.fr",null,null,null,null,[9,null,'
            '[["type.googleapis.com/assistant.boq.bard.application.BardErrorInfo",[1060]]]]]]\n'
        )

    def test_json_bard_error_raises(self):
        from gemini_web2api.gemini import extract_response_text
        with self.assertRaisesRegex(RuntimeError, "1060"):
            extract_response_text(self._json_error_payload())


class DynamicExitTest(unittest.TestCase):
    """S5: dynamic exits (rotate IP per request) must not be cooled down on failure."""

    def setUp(self):
        CONFIG["proxies"] = ["http://user:pass@127.0.0.1:2260", "socks5://exit-b:1080"]
        CONFIG["proxy_rotate"] = {
            "enabled": True,
            "cooldown_sec": 300,
            "fail_threshold": 1,
            "probe_on_start": False,
            "dynamic_exits": ["http://user:pass@127.0.0.1:2260"],
        }
        POOL.configure_from_config()

    def test_dynamic_exit_not_cooled_on_failure(self):
        POOL.mark_failure("http://user:pass@127.0.0.1:2260", "BardErrorInfo [1060]")
        status = POOL.status()
        dyn = [e for e in status["exits"] if e["proxy"] == "http://user:pass@127.0.0.1:2260"][0]
        self.assertTrue(dyn["available"], "dynamic exit must stay available (IP rotates per request)")
        self.assertEqual(dyn["cooldown_remaining_sec"], 0)
        self.assertEqual(status["current"], "socks5://exit-b:1080", "still rotates to next exit")

    def test_dynamic_exit_candidates_include_it_immediately(self):
        POOL.mark_failure("http://user:pass@127.0.0.1:2260", "rate limited")
        candidates = POOL.candidates()
        self.assertIn("http://user:pass@127.0.0.1:2260", candidates,
                      "dynamic exit returns to rotation immediately (new IP next request)")
