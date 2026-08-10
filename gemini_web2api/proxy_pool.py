"""Proxy pool with health tracking, cooldown, and automatic rotation."""
from __future__ import annotations

import threading
import time
from typing import Any, Optional
from urllib.error import HTTPError

from .config import CONFIG

# Errors that usually mean the current egress IP is blocked / captcha'd.
_BLOCK_STATUS = {302, 303, 307, 308, 403, 405, 429, 503}
_BLOCK_MARKERS = (
    "google.com/sorry",
    "www.google.com/sorry",
    "unusual traffic",
    "recaptcha",
    "our systems have detected",
    "sorry/index",
)


def log(msg: str):
    if CONFIG.get("log_requests"):
        import sys
        sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        sys.stderr.flush()


def _normalize_proxy(proxy: Optional[str]) -> Optional[str]:
    if proxy is None:
        return None
    proxy = str(proxy).strip()
    return proxy or None


def _proxy_label(proxy: Optional[str]) -> str:
    return proxy if proxy is not None else "direct"


class ProxyPool:
    """Thread-safe sticky proxy selection with failure-driven rotation."""

    def __init__(self):
        self._lock = threading.RLock()
        self._proxies: list[Optional[str]] = []
        self._state: dict[Optional[str], dict[str, Any]] = {}
        self._current: Optional[str] = None
        self._configured = False
        self._dynamic_exits: set = set()

    def configure_from_config(self) -> None:
        """Load proxy list from CONFIG. Safe to call multiple times."""
        proxies: list[Optional[str]] = []
        raw_list = CONFIG.get("proxies")
        if isinstance(raw_list, list) and raw_list:
            for item in raw_list:
                proxies.append(_normalize_proxy(item))
        else:
            single = _normalize_proxy(CONFIG.get("proxy"))
            proxies.append(single)

        # Preserve order, drop duplicates while keeping first occurrence.
        seen = set()
        ordered: list[Optional[str]] = []
        for p in proxies:
            key = _proxy_label(p)
            if key in seen:
                continue
            seen.add(key)
            ordered.append(p)

        rotate = CONFIG.get("proxy_rotate") or {}
        with self._lock:
            old_current = self._current
            self._proxies = ordered
            self._state = {
                p: self._state.get(p) or {
                    "fails": 0,
                    "cooldown_until": 0.0,
                    "last_ok": 0.0,
                    "last_fail": 0.0,
                    "last_error": None,
                    "successes": 0,
                    "failures": 0,
                }
                for p in ordered
            }
            if old_current in ordered:
                self._current = old_current
            else:
                self._current = ordered[0] if ordered else None
            self._configured = True
            self._rotate_cfg = {
                "enabled": bool(rotate.get("enabled", True)),
                "cooldown_sec": float(rotate.get("cooldown_sec", 300)),
                "fail_threshold": int(rotate.get("fail_threshold", 1)),
                "probe_on_start": bool(rotate.get("probe_on_start", False)),
            }
            self._dynamic_exits = {
                _normalize_proxy(p) for p in (rotate.get("dynamic_exits") or [])
            }

        labels = ", ".join(_proxy_label(p) for p in ordered) or "direct"
        log(f"Proxy pool: {len(ordered)} exit(s) [{labels}] current={_proxy_label(self._current)}")

    def enabled(self) -> bool:
        with self._lock:
            if not self._configured:
                self.configure_from_config()
            return bool(self._rotate_cfg.get("enabled", True)) and len(self._proxies) > 1

    def current(self) -> Optional[str]:
        with self._lock:
            if not self._configured:
                self.configure_from_config()
            return self._current

    def all_proxies(self) -> list[Optional[str]]:
        with self._lock:
            if not self._configured:
                self.configure_from_config()
            return list(self._proxies)

    def _cooldown_sec(self) -> float:
        return float(self._rotate_cfg.get("cooldown_sec", 300))

    def _fail_threshold(self) -> int:
        return int(self._rotate_cfg.get("fail_threshold", 1))

    def _is_available(self, proxy: Optional[str], now: float) -> bool:
        st = self._state.get(proxy)
        if not st:
            return True
        return now >= float(st.get("cooldown_until") or 0)

    def candidates(self, max_n: Optional[int] = None) -> list[Optional[str]]:
        """Return preferred proxy order: sticky current first, then other healthy, then cooled-down."""
        with self._lock:
            if not self._configured:
                self.configure_from_config()
            if not self._proxies:
                return [None]
            if not self._rotate_cfg.get("enabled", True) or len(self._proxies) == 1:
                return [self._current if self._current in self._proxies else self._proxies[0]]

            now = time.time()
            current = self._current if self._current in self._proxies else self._proxies[0]
            healthy: list[Optional[str]] = []
            cooled: list[Optional[str]] = []

            # Start from current index for round-robin among others.
            try:
                start = self._proxies.index(current)
            except ValueError:
                start = 0
            ordered = self._proxies[start:] + self._proxies[:start]

            for p in ordered:
                if self._is_available(p, now):
                    healthy.append(p)
                else:
                    cooled.append(p)

            # Prefer sticky current if healthy.
            result: list[Optional[str]] = []
            if current in healthy:
                result.append(current)
                result.extend(p for p in healthy if p != current)
            else:
                result.extend(healthy)

            # If everything is in cooldown, still try soonest-to-recover first.
            if not result:
                cooled_sorted = sorted(
                    cooled,
                    key=lambda p: float(self._state.get(p, {}).get("cooldown_until") or 0),
                )
                result = cooled_sorted

            if max_n is not None:
                result = result[:max_n]
            return result

    def mark_success(self, proxy: Optional[str]) -> None:
        with self._lock:
            if proxy not in self._state:
                return
            st = self._state[proxy]
            st["fails"] = 0
            st["cooldown_until"] = 0.0
            st["last_ok"] = time.time()
            st["successes"] = int(st.get("successes") or 0) + 1
            st["last_error"] = None
            self._current = proxy

    def mark_failure(self, proxy: Optional[str], reason: str, force_rotate: bool = True) -> Optional[str]:
        """Mark proxy failed. Returns the next proxy to try (may be same if rotation disabled)."""
        with self._lock:
            if not self._configured:
                self.configure_from_config()
            now = time.time()
            if proxy in self._state:
                st = self._state[proxy]
                st["fails"] = int(st.get("fails") or 0) + 1
                st["failures"] = int(st.get("failures") or 0) + 1
                st["last_fail"] = now
                st["last_error"] = reason[:300]
                if st["fails"] >= self._fail_threshold() and proxy not in self._dynamic_exits:
                    st["cooldown_until"] = now + self._cooldown_sec()
                    log(
                        f"Proxy cooldown {_proxy_label(proxy)} for {int(self._cooldown_sec())}s "
                        f"(fails={st['fails']}): {reason[:120]}"
                    )
                else:
                    log(f"Proxy fail {_proxy_label(proxy)} ({st['fails']}/{self._fail_threshold()}): {reason[:120]}")

            if not force_rotate or not self._rotate_cfg.get("enabled", True) or len(self._proxies) <= 1:
                return self._current

            # Rotate to next available exit different from failed one.
            nxt = None
            for p in self.candidates():
                if p != proxy:
                    nxt = p
                    break
            if nxt is None:
                # only one proxy
                nxt = proxy
            if nxt != self._current:
                log(f"Proxy rotate {_proxy_label(self._current)} -> {_proxy_label(nxt)}")
            self._current = nxt
            return nxt

    def status(self) -> dict:
        with self._lock:
            if not self._configured:
                self.configure_from_config()
            now = time.time()
            exits = []
            for p in self._proxies:
                st = self._state.get(p) or {}
                cd = float(st.get("cooldown_until") or 0)
                exits.append({
                    "proxy": _proxy_label(p),
                    "available": now >= cd,
                    "cooldown_remaining_sec": max(0, int(cd - now)),
                    "consecutive_fails": int(st.get("fails") or 0),
                    "successes": int(st.get("successes") or 0),
                    "failures": int(st.get("failures") or 0),
                    "last_ok": st.get("last_ok") or None,
                    "last_fail": st.get("last_fail") or None,
                    "last_error": st.get("last_error"),
                    "current": p == self._current,
                })
            return {
                "enabled": bool(self._rotate_cfg.get("enabled", True)),
                "cooldown_sec": self._cooldown_sec(),
                "fail_threshold": self._fail_threshold(),
                "current": _proxy_label(self._current),
                "exits": exits,
            }


POOL = ProxyPool()


def is_block_response(status_code: int, headers: Any = None, body: bytes | str = b"") -> bool:
    """Heuristic: Google captcha / rate-limit / method trap."""
    if status_code in _BLOCK_STATUS:
        # 302 alone is not always a block; check Location / body when possible.
        if status_code in (302, 303, 307, 308):
            loc = ""
            if headers is not None:
                try:
                    loc = headers.get("Location") or headers.get("location") or ""
                except Exception:
                    loc = ""
            text = loc
            if body:
                text += " " + (body.decode("utf-8", "replace") if isinstance(body, (bytes, bytearray)) else str(body))
            text_l = text.lower()
            if any(m in text_l for m in _BLOCK_MARKERS) or "sorry" in text_l:
                return True
            # Redirect away from StreamGenerate is suspicious for this API.
            if "streamgenerate" not in text_l and "gemini.google.com" not in text_l:
                return True
            return "google.com" in text_l and "sorry" in text_l
        return True

    if not body:
        return False
    sample = body[:4000]
    text = sample.decode("utf-8", "replace") if isinstance(sample, (bytes, bytearray)) else str(sample)
    text_l = text.lower()
    return any(m in text_l for m in _BLOCK_MARKERS)


def is_block_error(exc: BaseException, body: bytes | str = b"") -> bool:
    if isinstance(exc, HTTPError):
        headers = getattr(exc, "headers", None)
        if body == b"":
            try:
                body = exc.read()
            except Exception:
                body = b""
        return is_block_response(exc.code, headers, body)
    # httpx.HTTPStatusError
    resp = getattr(exc, "response", None)
    if resp is not None:
        try:
            content = body if body else resp.content
        except Exception:
            content = body
        return is_block_response(getattr(resp, "status_code", 0), getattr(resp, "headers", None), content)

    msg = str(exc).lower()
    if any(m in msg for m in ("405", "429", "method not allowed", "too many requests", "sorry")):
        return True
    return False


def error_reason(exc: BaseException) -> str:
    resp = getattr(exc, "response", None)
    if resp is not None:
        loc = ""
        try:
            loc = resp.headers.get("Location") or resp.headers.get("location") or ""
        except Exception:
            pass
        return f"HTTP {resp.status_code}" + (f" -> {loc[:120]}" if loc else "")
    if isinstance(exc, HTTPError):
        loc = ""
        try:
            loc = exc.headers.get("Location") if exc.headers else ""
        except Exception:
            pass
        return f"HTTP {exc.code}" + (f" -> {loc[:120]}" if loc else f": {exc.reason}")
    return f"{type(exc).__name__}: {exc}"
