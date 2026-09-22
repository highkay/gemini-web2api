"""Truth prober for sticky sessions: one real StreamGenerate POST per session.

Reuses the engine's own payload/URL/header builders -- this is the same request
the gateway makes, so a session that answers here answers for users.  Verdicts
follow the sticky-pool contract:

- ``ok``        - a text frame came back
- ``block``     - BardErrorInfo[n], a captcha/sorry redirect, or an answer-less body
- ``transport`` - tunnel/connection failure (no verdict about the egress IP)

The read budget is deliberately *wider* than the gateway's per-attempt budget
(25s vs 15s): every probe opens a cold tunnel, and cold resin sessions were
measured answering at 20-21s while warm reuse of the same session answers in 3-9s
(2026-09-23: 10 of 11 actives answered under a 40s budget, three of them just past
the old 15s cap).  A production attempt that meets a cold session may still time
out once -- the request then rotates to the next exit -- but the pool must not
evict a session that is merely cold.
"""
from __future__ import annotations

import time

import httpx

from .config import CONFIG
from .gemini import (
    _BARD_ERR_RE,
    _build_headers,
    _build_payload,
    _extract_texts_from_line,
    _get_url,
)
from .models import resolve_model
from .proxy_pool import is_block_response

try:
    from sticky_pool.types import ProbeResult
except ImportError:  # repo-side test runs without the sticky_pool module on sys.path
    from dataclasses import dataclass

    @dataclass
    class ProbeResult:  # type: ignore[no-redef]
        """Structural stand-in for sticky_pool.types.ProbeResult."""

        ok: bool
        signature: str
        latency_ms: float = 0.0
        detail: str = ""
        egress_ip: str = ""


def _client_factory(url: str, timeout: httpx.Timeout) -> httpx.Client:
    """Bind a client to one session URL.

    ``proxy=`` is httpx >= 0.26; the fallback keeps the prober usable on the
    httpx 0.23 that host-side test runs ship (the container has 0.28).
    """
    try:
        return httpx.Client(proxy=url, timeout=timeout, follow_redirects=False)
    except TypeError:
        return httpx.Client(proxies=url, timeout=timeout, follow_redirects=False)


def _result(ok: bool, signature: str, started: float, detail: str = "") -> ProbeResult:
    return ProbeResult(
        ok=ok,
        signature=signature,
        latency_ms=(time.monotonic() - started) * 1000.0,
        detail=detail,
    )


class GeminiStreamProber:
    """Probe one sticky session with a throwaway StreamGenerate request."""

    def __init__(
        self,
        prompt: str = "ping",
        model: str = None,
        connect_timeout: float = 8.0,
        read_timeout: float = 25.0,
        write_timeout: float = 10.0,
        pool_timeout: float = 5.0,
    ):
        default_model = model or CONFIG.get("default_model") or "gemini-3.6-flash"
        _name, model_id, think_mode, error, extra = resolve_model(default_model)
        if error:
            raise ValueError(f"GeminiStreamProber: {error}")
        self.prompt = prompt
        self.model_id = model_id
        self.think_mode = think_mode
        self.extra_fields = extra
        self.timeout = httpx.Timeout(
            connect=connect_timeout, read=read_timeout, write=write_timeout, pool=pool_timeout
        )

    def __call__(self, url: str) -> ProbeResult:
        started = time.monotonic()
        body = _build_payload(
            self.prompt, self.model_id, self.think_mode, None, self.extra_fields
        ).encode()
        try:
            with _client_factory(url, self.timeout) as client:
                resp = client.post(_get_url(), content=body, headers=_build_headers())
                status = resp.status_code
                headers = resp.headers
                content = resp.content
        except Exception as exc:  # noqa: BLE001 - every failure here is a transport verdict
            return _result(False, "transport", started, f"{type(exc).__name__}: {exc}"[:200])

        text = content.decode("utf-8", errors="replace")
        bard = _BARD_ERR_RE.search(text)
        if bard:
            return _result(False, "block", started, f"BardErrorInfo [{bard.group(1)}]")
        if is_block_response(status, headers, content):
            location = ""
            try:
                location = headers.get("Location") or ""
            except Exception:  # noqa: BLE001
                pass
            return _result(
                False, "block", started,
                f"http={status} bytes={len(content)}" + (f" loc={location[:80]}" if location else ""),
            )
        if status >= 400:
            return _result(False, "transport", started, f"http={status} bytes={len(content)}")

        answer = ""
        for line in text.split("\n"):
            for candidate in _extract_texts_from_line(line):
                if len(candidate) > len(answer):
                    answer = candidate
        if not answer:
            # HTTP 200 without an answer frame: the exit accepted the request but
            # produced nothing usable -- not a session to hand out.
            return _result(False, "block", started, f"no-text http={status} bytes={len(content)}")
        return _result(True, "ok", started, answer[:60].replace("\n", " "))