"""Sticky-screened pool wiring in the gateway: proxy pool injection + truth prober.

Covers two contracts the deployment depends on:
1. ProxyPool consumes the sticky-pool state file: a rotating, size-capped slice of
   the file's active sessions joins the exit list as dynamic exits, without
   growing the pool (and therefore the per-request attempt budget) on reloads.
2. GeminiStreamProber classifies a StreamGenerate response the same way the engine
   does (answer / BardErrorInfo / captcha redirect / tunnel failure).
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx

from gemini_web2api import sticky_gemini_prober
from gemini_web2api.config import CONFIG
from gemini_web2api.gemini import _max_proxy_attempts, _proxy_read_timeout, reset_httpx_clients
from gemini_web2api.proxy_pool import POOL
from gemini_web2api.sticky_gemini_prober import GeminiStreamProber

STATICS = ["socks5://192.168.112.1:7890", "socks5://192.168.112.1:7892"]


def sticky_url(n):
    return f"http://Default.wt22.user{n}:pw@192.168.1.18:2260"


def write_state(path, states, mtime=None, ips=None):
    """states: {n: state}.  mtime forces a distinct mtime (reload key)."""
    sessions = [
        {"n": n, "url": sticky_url(n), "state": state, "last_ok": 1.0, "last_check": 1.0,
         "consecutive_fails": 0, "egress_ip": (ips or {}).get(n, ""), "last_used": 0.0}
        for n, state in sorted(states.items())
    ]
    with open(path, "w") as fh:
        json.dump({"base": "Default.wt22", "host": "192.168.1.18", "port": 2260,
                   "sessions": sessions}, fh)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


class StickyWiringTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.state = os.path.join(self.dir, "sticky.json")
        CONFIG["proxies"] = list(STATICS)
        CONFIG["proxy_rotate"] = {"enabled": True, "cooldown_sec": 60, "fail_threshold": 2}
        CONFIG.pop("sticky_pool", None)
        reset_httpx_clients()
        POOL.configure_from_config()
        self._tick = 1000

    def tearDown(self):
        CONFIG.pop("sticky_pool", None)
        CONFIG["proxies"] = []
        reset_httpx_clients()
        shutil.rmtree(self.dir, ignore_errors=True)

    def _enable(self, **cfg):
        CONFIG["sticky_pool"] = {"enabled": True, "state_path": self.state,
                                 "max_dynamic_exits": 3, **cfg}
        POOL.configure_from_config()

    def _reload(self):
        """Rewrite the state file with a fresh mtime; the pool picks it up on demand."""
        self._tick += 10
        os.utime(self.state, (self._tick, self._tick))
        return POOL.candidates()

    def _exit_urls(self):
        return [p for p in POOL.all_proxies() if p is not None]

    def test_active_sessions_join_the_pool_as_dynamic_exits(self):
        write_state(self.state, {1: "active", 2: "active", 3: "active", 4: "evicted"})
        self._enable()

        self.assertEqual(self._exit_urls(), STATICS + [sticky_url(1), sticky_url(2), sticky_url(3)])
        self.assertEqual(_max_proxy_attempts(), 5, "2 statics + capped sticky slice, not +all")
        self.assertTrue(POOL.is_dynamic(sticky_url(1)))
        self.assertFalse(POOL.is_dynamic(STATICS[0]))
        status = POOL.status()
        self.assertEqual(status["sticky"]["injected"], 3)
        self.assertEqual(status["sticky"]["active_total"], 3)
        sticky_exits = [e for e in status["exits"] if e["sticky"]]
        self.assertEqual(len(sticky_exits), 3)
        self.assertTrue(all(e["dynamic"] for e in sticky_exits))
        self.assertEqual(POOL.candidates()[:2], STATICS, "static tunnels still lead")

    def test_slice_rotates_across_reloads_without_growing_the_pool(self):
        write_state(self.state, {n: "active" for n in range(1, 9)})
        self._enable()
        counts = {}
        slices = []

        def take_slice():
            urls = [u for u in self._exit_urls() if u not in STATICS]
            slices.append(urls)
            for url in urls:
                counts[url] = counts.get(url, 0) + 1
            self.assertEqual(len(self._exit_urls()), 5,
                             "reloading must replace the sticky slice, never append to it")

        take_slice()
        self.assertEqual(slices[0], [sticky_url(1), sticky_url(2), sticky_url(3)])
        for _ in range(7):
            self._reload()
            take_slice()

        self.assertEqual(slices[1], [sticky_url(4), sticky_url(5), sticky_url(6)])
        self.assertEqual(slices[2], [sticky_url(7), sticky_url(8), sticky_url(1)])
        self.assertEqual(counts, {sticky_url(n): 3 for n in range(1, 9)},
                         "reloads must spread load evenly over the active set")
        self.assertEqual(_max_proxy_attempts(), 5)

    def test_evicted_session_leaves_the_pool_on_reload(self):
        write_state(self.state, {1: "active", 2: "active", 3: "active", 4: "evicted"})
        self._enable(max_dynamic_exits=8)
        self.assertEqual(self._exit_urls(), STATICS + [sticky_url(n) for n in (1, 2, 3)])

        write_state(self.state, {1: "active", 2: "active", 3: "evicted", 4: "evicted"},
                    mtime=self._tick + 10)
        self._tick += 10
        POOL.candidates()

        self.assertEqual(self._exit_urls(), STATICS + [sticky_url(1), sticky_url(2)])
        self.assertNotIn(sticky_url(3), POOL.status()["exits"][-1].values())

    def test_empty_screened_pool_clears_the_dynamic_layer(self):
        write_state(self.state, {1: "active", 2: "active"})
        self._enable()
        self.assertEqual(len(self._exit_urls()), 4)

        write_state(self.state, {1: "evicted", 2: "evicted"}, mtime=self._tick + 10)
        self._tick += 10
        POOL.candidates()

        self.assertEqual(self._exit_urls(), STATICS)
        self.assertEqual(POOL.status()["sticky"]["injected"], 0)

    def test_slice_prefers_distinct_egress_ips(self):
        write_state(self.state, {1: "active", 2: "active", 3: "active", 4: "active"},
                    ips={1: "203.0.113.1", 2: "203.0.113.1", 3: "203.0.113.2",
                         4: "203.0.113.3"})
        self._enable()

        self.assertEqual([u for u in self._exit_urls() if u not in STATICS],
                         [sticky_url(1), sticky_url(3), sticky_url(4)],
                         "two sessions behind one egress IP must not take two attempt slots")

    def test_slice_backfills_when_ips_cannot_be_diversified(self):
        write_state(self.state, {1: "active", 2: "active", 3: "active"},
                    ips={1: "203.0.113.9", 2: "203.0.113.9", 3: "203.0.113.9"})
        self._enable()

        self.assertEqual(len([u for u in self._exit_urls() if u not in STATICS]), 3,
                         "a shared egress IP is a preference, not a reason to shrink the slice")

    def test_config_dynamic_exits_survive_sticky_reloads(self):
        legacy = "http://legacy:pass@127.0.0.1:2260"
        CONFIG["proxies"] = list(STATICS) + [legacy]
        CONFIG["proxy_rotate"]["dynamic_exits"] = [legacy]
        write_state(self.state, {1: "active", 2: "active"})
        self._enable()

        self.assertIn(legacy, self._exit_urls())
        self.assertTrue(POOL.is_dynamic(legacy))

        self._reload()

        self.assertIn(legacy, self._exit_urls(),
                      "a config-declared dynamic exit must not be dropped by the sticky slice")
        self.assertIn(legacy, POOL.candidates())

    def test_missing_state_file_keeps_the_statics_only(self):
        self._enable()
        self.assertEqual(self._exit_urls(), STATICS)
        self.assertEqual(POOL.status()["sticky"]["active_total"], 0)

    def test_status_reflects_a_state_change_without_a_chat_request(self):
        write_state(self.state, {1: "active", 2: "active"})
        self._enable()
        self.assertEqual(POOL.status()["sticky"]["injected"], 2)

        write_state(self.state, {1: "active", 2: "active", 3: "active"},
                    mtime=self._tick + 10)
        self._tick += 10

        status = POOL.status()

        self.assertEqual(status["sticky"]["injected"], 3,
                         "/status is the ops window: it must show a freshly written pool")
        self.assertEqual(len(status["exits"]), 5)

    def test_corrupt_state_file_keeps_the_working_slice(self):
        write_state(self.state, {1: "active", 2: "active"})
        self._enable()
        injected = self._exit_urls()

        with open(self.state, "w") as fh:
            fh.write("{not json")
        self._reload()

        self.assertEqual(self._exit_urls(), injected,
                         "an unreadable state file must not clear exits in use")

    def test_sticky_exit_is_available_immediately_after_failures(self):
        write_state(self.state, {1: "active", 2: "active"})
        self._enable()
        POOL.mark_failure(sticky_url(1), "blocked HTTP 302")
        POOL.mark_failure(sticky_url(1), "blocked HTTP 302")

        exit_state = [e for e in POOL.status()["exits"] if e["proxy"] == sticky_url(1)][0]
        self.assertTrue(exit_state["available"], "a rotating exit must never be cooled down")
        self.assertEqual(exit_state["cooldown_remaining_sec"], 0)
        self.assertEqual(exit_state["consecutive_fails"], 2)
        self.assertEqual(POOL.status()["current"], STATICS[0])
        self.assertEqual(POOL.candidates()[:2], STATICS)
        self.assertIn(sticky_url(2), POOL.candidates())

    def test_sticky_exit_drops_out_of_the_pool_when_it_retires(self):
        write_state(self.state, {1: "active", 2: "active"})
        self._enable()

        write_state(self.state, {1: "active", 2: "blocked"}, mtime=self._tick + 10)
        self._tick += 10
        POOL.candidates()

        self.assertNotIn(sticky_url(2), POOL.all_proxies())
        self.assertEqual([e["proxy"] for e in POOL.status()["exits"]], STATICS + [sticky_url(1)])

    def test_read_budget_is_shorter_for_sticky_exits(self):
        write_state(self.state, {1: "active"})
        self._enable()

        self.assertEqual(_proxy_read_timeout(sticky_url(1)), 15.0)
        self.assertEqual(_proxy_read_timeout(STATICS[0]), 30.0)
        self.assertEqual(_proxy_read_timeout(None), 30.0)


class ProberTest(unittest.TestCase):
    """GeminiStreamProber verdicts; the HTTP layer is replaced by a mock transport."""

    def _probe(self, handler):
        def factory(url, timeout):
            return httpx.Client(transport=httpx.MockTransport(handler), timeout=timeout)

        original = sticky_gemini_prober._client_factory
        sticky_gemini_prober._client_factory = factory
        try:
            return GeminiStreamProber()(sticky_url(1))
        finally:
            sticky_gemini_prober._client_factory = original

    def _answer_body(self, text="pong from the screened session"):
        # Real wrb.fr lines are long; the engine's parser ignores lines under 200 chars.
        inner = [None, None, None, None, [[None, [text]]], None, None, None, ["pad" * 60]]
        frame = json.dumps([["wrb.fr", None, json.dumps(inner)]])
        return ")]}'\n\n121\n" + frame + "\n"

    def test_answer_frame_is_ok(self):
        result = self._probe(lambda request: httpx.Response(200, text=self._answer_body()))
        self.assertTrue(result.ok, result.detail)
        self.assertEqual(result.signature, "ok")
        self.assertIn("pong", result.detail)
        self.assertGreaterEqual(result.latency_ms, 0)
        self.assertEqual(result.egress_ip, "")

    def test_bard_error_info_is_block(self):
        body = (
            ")]}'\n\n121\n"
            '[["wrb.fr",null,null,null,null,[9,null,'
            '[["type.googleapis.com/assistant.boq.bard.application.BardErrorInfo",[1060]]]]]]\n'
        )
        result = self._probe(lambda request: httpx.Response(200, text=body))
        self.assertFalse(result.ok)
        self.assertEqual(result.signature, "block")
        self.assertIn("1060", result.detail)

    def test_captcha_redirect_is_block(self):
        def handler(request):
            return httpx.Response(
                302, headers={"Location": "https://www.google.com/sorry/index?continue=x"},
                content=b"s" * 532,
            )

        result = self._probe(handler)
        self.assertFalse(result.ok)
        self.assertEqual(result.signature, "block")
        self.assertIn("http=302", result.detail)

    def test_answer_less_body_is_block(self):
        result = self._probe(lambda request: httpx.Response(200, text=")]}'\n\n121\n"))
        self.assertFalse(result.ok)
        self.assertEqual(result.signature, "block")
        self.assertIn("no-text", result.detail)

    def test_tunnel_failure_is_transport(self):
        def handler(request):
            raise httpx.ConnectError("tunnel down")

        result = self._probe(handler)
        self.assertFalse(result.ok)
        self.assertEqual(result.signature, "transport")
        self.assertIn("ConnectError", result.detail)

    def test_server_error_is_transport(self):
        result = self._probe(lambda request: httpx.Response(502, text="bad gateway"))
        self.assertFalse(result.ok)
        self.assertEqual(result.signature, "transport")
        self.assertIn("http=502", result.detail)

    def test_probe_budget_mirrors_production(self):
        """A probe that waits longer than production keeps sessions production can't use."""
        prober = GeminiStreamProber()
        self.assertEqual(prober.timeout.connect, 8.0)
        self.assertEqual(prober.timeout.read, 15.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)