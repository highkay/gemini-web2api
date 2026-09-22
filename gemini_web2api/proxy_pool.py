"""Proxy pool with health tracking, cooldown, and automatic rotation."""
from __future__ import annotations

import json
import os
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


def _fresh_state() -> dict[str, Any]:
    return {
        "fails": 0,
        "cooldown_until": 0.0,
        "last_ok": 0.0,
        "last_fail": 0.0,
        "last_error": None,
        "successes": 0,
        "failures": 0,
    }


def _read_sticky_active_urls(state_path: str) -> Optional[list[tuple[str, str]]]:
    """``[(url, egress_ip), ...]`` for the active sessions; ``None`` = unreadable.

    The sticky_pool module owns this format; the gateway reads only the fields it needs
    so a missing or corrupt state file degrades to statics-only instead of taking the
    gateway down.
    """
    try:
        with open(state_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        sessions = data.get("sessions") or []
    except (OSError, ValueError, AttributeError, TypeError):
        return None
    active = []
    for session in sessions:
        if isinstance(session, dict) and session.get("state") == "active" and session.get("url"):
            active.append((str(session["url"]), str(session.get("egress_ip") or "")))
    return active


def _prefer_distinct_ips(entries: list[tuple[str, str]], count: int) -> list[str]:
    """Pick ``count`` URLs, preferring one per egress IP (then backfilling).

    Sticky sessions can share an egress IP (two sessions behind one flagged exit), and
    a slice that spends two of five attempts on the same IP gets one draw, not two.
    ``egress_ip`` is only as fresh as the last screen, so this is a preference: the
    backfill keeps the slice at full size and never drops a usable session.
    """
    picked: list[str] = []
    seen_ips: set[str] = set()
    for url, ip in entries:
        if ip and ip in seen_ips:
            continue
        if ip:
            seen_ips.add(ip)
        picked.append(url)
        if len(picked) >= count:
            return picked
    for url, _ip in entries:
        if url not in picked:
            picked.append(url)
            if len(picked) >= count:
                break
    return picked


class ProxyPool:
    """Thread-safe sticky proxy selection with failure-driven rotation."""

    def __init__(self):
        self._lock = threading.RLock()
        self._proxies: list[Optional[str]] = []
        self._state: dict[Optional[str], dict[str, Any]] = {}
        self._current: Optional[str] = None
        self._configured = False
        self._dynamic_exits: set = set()
        self._config_dynamic_exits: set = set()
        self._config_dynamic_list: list = []
        # Screened sticky pool (see _load_sticky): the CLI re-screens resin-style
        # sessions and publishes the handout-able ones; the gateway injects a
        # rotating slice of them as dynamic exits whenever the file changes.
        self._sticky_cfg: dict = {}
        self._sticky_urls: list[str] = []
        self._sticky_active_total = 0
        self._sticky_offset = 0
        self._sticky_state_mtime = 0.0
        self._client_evictor = None

    def set_client_evictor(self, fn) -> None:
        """Register ``fn([url, ...])``: close cached clients for retired exits.

        Called with POOL's lock held, so ``fn`` may only take locks that are never
        taken while reaching for POOL's lock.  gemini.py's evictor takes the httpx
        client lock, and the holder of that lock resolves its timeouts before
        acquiring it -- keep that order (POOL -> _httpx, never the reverse).
        """
        self._client_evictor = fn

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
            self._state = {p: self._state.get(p) or _fresh_state() for p in ordered}
            self._dynamic_exits = {
                _normalize_proxy(p) for p in (rotate.get("dynamic_exits") or [])
            }
            self._config_dynamic_exits = set(self._dynamic_exits)
            self._config_dynamic_list = [
                _normalize_proxy(p) for p in (rotate.get("dynamic_exits") or [])
                if _normalize_proxy(p)
            ]
            # Dynamic exits (per-request IP rotation) are a last-resort layer, never a
            # place to pin the pool: a single fallback to one of them must not make every
            # later request lead with it (2026-09-22).
            static_ordered = [p for p in ordered if p not in self._dynamic_exits]
            if old_current in ordered and old_current not in self._dynamic_exits:
                self._current = old_current
            else:
                self._current = (static_ordered or ordered or [None])[0]
            self._configured = True
            self._rotate_cfg = {
                "enabled": bool(rotate.get("enabled", True)),
                "cooldown_sec": float(rotate.get("cooldown_sec", 300)),
                "fail_threshold": int(rotate.get("fail_threshold", 1)),
                "probe_on_start": bool(rotate.get("probe_on_start", False)),
            }
            sticky_cfg = CONFIG.get("sticky_pool")
            self._sticky_cfg = dict(sticky_cfg) if isinstance(sticky_cfg, dict) else {}
            self._sticky_urls = []
            self._sticky_active_total = 0
            self._sticky_offset = 0
            self._sticky_state_mtime = 0.0

        labels = ", ".join(_proxy_label(p) for p in ordered) or "direct"
        log(f"Proxy pool: {len(ordered)} exit(s) [{labels}] current={_proxy_label(self._current)}")
        if self._sticky_cfg.get("enabled"):
            self._load_sticky(force=True)

    def enabled(self) -> bool:
        with self._lock:
            if not self._configured:
                self.configure_from_config()
            return bool(self._rotate_cfg.get("enabled", True)) and len(self._proxies) > 1

    def _load_sticky(self, force: bool = False) -> bool:
        """Inject the screened sticky sessions published by the sticky-pool CLI.

        Only ``max_dynamic_exits`` of the file's active sessions are injected and the
        slice rotates on every reload: ``_max_proxy_attempts()`` is
        ``max(retry_attempts, len(_proxies))``, so every injected URL adds another
        attempt to a failing request.  With 2 static tunnels + 3 sticky exits the
        observed failure chain is ~72s, inside the 120s caller budget at :9010;
        injecting all ~20 screened sessions would put a failing request over 6 minutes.

        Cheap and safe to call per request: it is a stat() plus an mtime compare, and
        it changes nothing until the CLI writes a new state file.
        """
        cfg = self._sticky_cfg
        if not cfg.get("enabled"):
            return False
        state_path = cfg.get("state_path")
        if not state_path:
            return False
        try:
            mtime = os.stat(state_path).st_mtime
        except OSError:
            return False  # before the first `init`: statics-only, nothing to change

        with self._lock:
            if not force and mtime == self._sticky_state_mtime:
                return False
            self._sticky_state_mtime = mtime
            active_entries = _read_sticky_active_urls(state_path)
            if active_entries is None:
                log(f"Sticky pool: {state_path} unreadable; keeping current exits")
                return False

            max_dynamic = max(1, int(cfg.get("max_dynamic_exits", 3)))
            if len(active_entries) > max_dynamic:
                offset = self._sticky_offset % len(active_entries)
                rotated = active_entries[offset:] + active_entries[:offset]
                urls = _prefer_distinct_ips(rotated, max_dynamic)
                self._sticky_offset = (offset + max_dynamic) % len(active_entries)
            else:
                urls = _prefer_distinct_ips(active_entries, max_dynamic)
                self._sticky_offset = 0

            previous_dynamic = set(self._dynamic_exits)
            statics = [p for p in self._proxies if p not in previous_dynamic]
            self._dynamic_exits = set(self._config_dynamic_exits) | set(urls)
            # Replace (never append) the sticky slice, but keep config-declared dynamic
            # exits: this runs on every file change, so appending would grow _proxies --
            # and the attempt budget -- forever.
            self._proxies = statics + [
                p for p in self._config_dynamic_list if p not in statics
            ] + [p for p in urls if p not in statics and p not in self._config_dynamic_exits]
            self._sticky_urls = list(urls)
            self._sticky_active_total = len(active_entries)
            for proxy in self._proxies:
                self._state.setdefault(proxy, _fresh_state())
            # Exits that left the pool must not keep counters around: /status and the
            # cooldown accounting read exactly this map.
            for proxy in [p for p in self._state if p not in self._proxies]:
                self._state.pop(proxy, None)
            static_ordered = [p for p in self._proxies if p not in self._dynamic_exits]
            if self._current not in self._proxies or self._current in self._dynamic_exits:
                self._current = (static_ordered or self._proxies or [None])[0]
            retired = sorted(p for p in previous_dynamic - self._dynamic_exits if p)

        if retired and self._client_evictor is not None:
            try:
                self._client_evictor(retired)
            except Exception:  # pragma: no cover - eviction must never break a request
                pass
        log(
            f"Sticky pool: {len(urls)} exit(s) injected of {len(active_entries)} active "
            f"[{', '.join(_proxy_label(p) for p in urls) or 'none'}]"
        )
        return True

    def is_dynamic(self, proxy: Optional[str]) -> bool:
        """True for per-request-rotating exits (screened sticky pool or config)."""
        with self._lock:
            if not self._configured:
                self.configure_from_config()
            return proxy in self._dynamic_exits

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
        # Pick up a freshly screened sticky pool within one request of the CLI writing it.
        self._load_sticky()
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

            # Prefer sticky current if healthy; dynamic exits always trail the statics.
            healthy_static = [p for p in healthy if p not in self._dynamic_exits]
            healthy_dynamic = [p for p in healthy if p in self._dynamic_exits]
            result: list[Optional[str]] = []
            if current in healthy_static:
                result.append(current)
                result.extend(p for p in healthy_static if p != current)
            else:
                result.extend(healthy_static)
            result.extend(healthy_dynamic)

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
            if proxy not in self._dynamic_exits:
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
        # /status is the ops window onto this pool: pick up a state-file change here too,
        # so a re-screened pool is visible without waiting for a chat request.
        self._load_sticky()
        with self._lock:
            if not self._configured:
                self.configure_from_config()
            now = time.time()
            exits = []
            sticky_urls = set(self._sticky_urls)
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
                    "dynamic": p in self._dynamic_exits,
                    "sticky": p in sticky_urls,
                })
            return {
                "enabled": bool(self._rotate_cfg.get("enabled", True)),
                "cooldown_sec": self._cooldown_sec(),
                "fail_threshold": self._fail_threshold(),
                "current": _proxy_label(self._current),
                "exits": exits,
                "sticky": {
                    "enabled": bool(self._sticky_cfg.get("enabled")),
                    "state_path": self._sticky_cfg.get("state_path"),
                    "state_mtime": self._sticky_state_mtime or None,
                    "max_dynamic_exits": self._sticky_cfg.get("max_dynamic_exits"),
                    "injected": len(self._sticky_urls),
                    "active_total": self._sticky_active_total,
                },
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
