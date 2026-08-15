"""Gemini StreamGenerate protocol implementation with httpx streaming."""
import json
import time
import uuid
import re
import urllib.request
import urllib.parse
import urllib.error
import ssl
import os
import hashlib
from typing import Optional

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

from .config import CONFIG
from .proxy_pool import POOL, is_block_error, is_block_response, error_reason

_ssl_ctx = None
_cookie_cache = {"str": "", "sapisid": None, "mtime": 0}
_httpx_clients: dict[str, "httpx.Client"] = {}
_httpx_lock = None


def log(msg: str):
    if CONFIG["log_requests"]:
        import sys
        sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        sys.stderr.flush()


def _get_ssl_ctx():
    global _ssl_ctx
    if _ssl_ctx is None:
        _ssl_ctx = ssl.create_default_context()
    return _ssl_ctx


def _client_key(proxy: Optional[str]) -> str:
    return proxy or "__direct__"


def _get_httpx_client(proxy: Optional[str] = None):
    """Get or create a per-proxy httpx client (supports socks5:// via socksio)."""
    global _httpx_lock
    if not HAS_HTTPX:
        return None
    if _httpx_lock is None:
        import threading
        _httpx_lock = threading.Lock()
    key = _client_key(proxy)
    with _httpx_lock:
        client = _httpx_clients.get(key)
        if client is not None:
            return client
        # Follow redirects only for non-API pages; StreamGenerate captcha is 302.
        # We disable auto-follow so we can detect block redirects cleanly.
        timeout = CONFIG["request_timeout_sec"]
        if proxy:
            client = httpx.Client(
                proxy=proxy,
                timeout=timeout,
                verify=True,
                follow_redirects=False,
            )
        else:
            client = httpx.Client(
                timeout=timeout,
                verify=True,
                follow_redirects=False,
            )
        _httpx_clients[key] = client
        return client


def reset_httpx_clients():
    """Close cached clients (e.g. after config reload)."""
    global _httpx_clients
    for c in list(_httpx_clients.values()):
        try:
            c.close()
        except Exception:
            pass
    _httpx_clients = {}


def load_cookie() -> tuple:
    """Load cookie from file with mtime-based caching."""
    cookie_file = CONFIG.get("cookie_file")
    if not cookie_file or not os.path.exists(cookie_file):
        return "", None
    try:
        mtime = os.path.getmtime(cookie_file)
        if mtime == _cookie_cache["mtime"] and _cookie_cache["str"]:
            return _cookie_cache["str"], _cookie_cache["sapisid"]
        with open(cookie_file, "r") as f:
            content = f.read().strip()
        if content.startswith("{"):
            data = json.loads(content)
            cookie_str = data.get("cookie", "")
            sapisid = data.get("sapisid", "")
        else:
            cookie_str = content
            pairs = dict(p.split("=", 1) for p in cookie_str.split("; ") if "=" in p)
            sapisid = pairs.get("SAPISID", "")
        _cookie_cache.update({"str": cookie_str, "sapisid": sapisid or None, "mtime": mtime})
        return cookie_str, sapisid if sapisid else None
    except Exception as e:
        log(f"Cookie load error: {e}")
        return _cookie_cache["str"], _cookie_cache["sapisid"]


def make_sapisidhash(sapisid: str) -> str:
    ts = int(time.time())
    h = hashlib.sha1(f"{ts} {sapisid} https://gemini.google.com".encode()).hexdigest()
    return f"SAPISIDHASH {ts}_{h}"


def _account_prefix() -> str:
    """Return the Gemini account path prefix for non-default Google accounts."""
    auth_user = CONFIG.get("auth_user")
    if auth_user is None or auth_user == "":
        return ""
    return f"/u/{auth_user}"


def _build_headers() -> dict:
    account_prefix = _account_prefix()
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Origin": "https://gemini.google.com",
        "Referer": f"https://gemini.google.com{account_prefix}/app",
        "X-Same-Domain": "1",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    if account_prefix:
        headers["X-Goog-AuthUser"] = str(CONFIG["auth_user"])
    cookie_str, sapisid = load_cookie()
    if cookie_str:
        headers["Cookie"] = cookie_str
    if sapisid:
        headers["Authorization"] = make_sapisidhash(sapisid)
    return headers


def _apply_chat_persistence_flags(inner: list) -> None:
    """Apply Gemini Web persistence flags to an outgoing request payload."""
    if CONFIG.get("temporary_chats", False):
        # Match Gemini Web temporary-chat requests.
        inner[41] = [1]
        inner[45] = 1
    else:
        inner[41] = [2]


def _build_payload(prompt: str, model_id: int, think_mode: int, file_refs: list = None, extra_fields: dict = None) -> str:
    inner = [None] * 102
    if file_refs:
        refs = [[None, None, ref] for ref in file_refs]
        inner[0] = [prompt, 0, None, refs, None, None, 0]
    else:
        inner[0] = [prompt, 0, None, None, None, None, 0]
    inner[1] = ["en"]
    inner[2] = ["", "", "", None, None, None, None, None, None, ""]
    inner[6] = [0]
    inner[7] = 1
    inner[10] = 1
    inner[11] = 0
    inner[17] = [[think_mode]]
    inner[18] = 0
    inner[27] = 1
    inner[30] = [4]
    _apply_chat_persistence_flags(inner)
    inner[53] = 0
    inner[59] = str(uuid.uuid4())
    inner[61] = []
    inner[68] = 1
    inner[79] = model_id
    if extra_fields:
        for k, v in extra_fields.items():
            inner[k] = v
    outer = [None, json.dumps(inner)]
    params = {"f.req": json.dumps(outer)}
    if CONFIG.get("xsrf_token"):
        params["at"] = CONFIG["xsrf_token"]
    return urllib.parse.urlencode(params)


def _get_url() -> str:
    reqid = int(time.time()) % 1000000
    account_prefix = _account_prefix()
    return (
        f"https://gemini.google.com{account_prefix}/_/BardChatUi/data/"
        "assistant.lamda.BardFrontendService/StreamGenerate"
        f"?bl={CONFIG['gemini_bl']}&hl=en&_reqid={reqid}&rt=c"
    )


def clean_text(text: str, strip: bool = True) -> str:
    text = re.sub(
        r'```(?:python|javascript|text)\?code_(?:reference|stdout)&code_event_index=\d+\n.*?```\n?',
        '', text, flags=re.DOTALL
    )
    text = re.sub(r'http://googleusercontent\.com/card_content/\d+\n?', '', text)
    return text.strip() if strip else text


def _extract_texts_from_line(line: str) -> list:
    """Parse a single wrb.fr line and return list of text strings found."""
    if '"wrb.fr"' not in line or len(line) < 200:
        return []
    try:
        arr = json.loads(line)
        inner_str = arr[0][2]
        if not inner_str or len(inner_str) < 50:
            return []
        inner = json.loads(inner_str)
        if not (isinstance(inner, list) and len(inner) > 4 and inner[4]):
            return []
        texts = []
        for part in inner[4]:
            if isinstance(part, list) and len(part) > 1 and part[1] and isinstance(part[1], list):
                for t in part[1]:
                    if isinstance(t, str) and t:
                        texts.append(t)
        return texts
    except (json.JSONDecodeError, IndexError, TypeError):
        return []


_BARD_ERR_RE = re.compile(r'BardErrorInfo[^\d]*\[(\d+)\]')


def _raise_bard_error(raw: str):
    bard_err = _BARD_ERR_RE.search(raw)
    if bard_err:
        raise RuntimeError(f"Gemini upstream rejected request: BardErrorInfo [{bard_err.group(1)}]")


def extract_response_text(raw: str) -> str:
    """Parse full response to get final text."""
    _raise_bard_error(raw)
    last_text = ""
    for line in raw.split("\n"):
        for t in _extract_texts_from_line(line):
            if len(t) > len(last_text):
                last_text = t
    return clean_text(last_text)


def _max_proxy_attempts() -> int:
    proxies = POOL.all_proxies()
    # Try each exit once per request, capped by retry_attempts * pool size.
    return max(CONFIG["retry_attempts"], len(proxies) if proxies else 1)


class BlockedUpstreamError(RuntimeError):
    """Raised when Google returns captcha / rate-limit style block."""
    def __init__(self, message: str, status_code: int = 0):
        super().__init__(message)
        self.status_code = status_code


def _open_with_proxy(req: urllib.request.Request, proxy: Optional[str], timeout: float):
    """Open request via optional HTTP(S) proxy using urllib. SOCKS not supported here."""
    ctx = _get_ssl_ctx()
    if proxy and proxy.startswith(("socks5://", "socks5h://", "socks4://")):
        raise RuntimeError(f"urllib path does not support SOCKS proxy: {proxy} (use httpx)")
    if proxy:
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy, "https": proxy}),
            urllib.request.HTTPSHandler(context=ctx),
        )
        # Do not auto-follow redirects — captcha is a 302.
        class _NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None
        opener.add_handler(_NoRedirect())
        return opener.open(req, timeout=timeout)
    # direct
    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None
    opener = urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=ctx),
        _NoRedirect(),
    )
    return opener.open(req, timeout=timeout)


def _generate_once_httpx(url: str, body: bytes, headers: dict, proxy: Optional[str]) -> str:
    client = _get_httpx_client(proxy)
    resp = client.post(url, content=body, headers=headers)
    content = resp.content
    if is_block_response(resp.status_code, resp.headers, content):
        loc = resp.headers.get("Location") or resp.headers.get("location") or ""
        raise BlockedUpstreamError(
            f"blocked HTTP {resp.status_code}" + (f" -> {loc[:120]}" if loc else ""),
            status_code=resp.status_code,
        )
    resp.raise_for_status()
    raw = content.decode("utf-8", errors="replace")
    if is_block_response(200, None, raw):
        raise BlockedUpstreamError("blocked body (captcha/sorry)")
    return extract_response_text(raw)


def _generate_once_urllib(url: str, body: bytes, headers: dict, proxy: Optional[str]) -> str:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        resp = _open_with_proxy(req, proxy, CONFIG["request_timeout_sec"])
        raw_bytes = resp.read()
        raw = raw_bytes.decode("utf-8", errors="replace")
        if is_block_response(getattr(resp, "status", 200), resp.headers, raw_bytes):
            raise BlockedUpstreamError("blocked response")
        return extract_response_text(raw)
    except urllib.error.HTTPError as e:
        body_bytes = b""
        try:
            body_bytes = e.read()
        except Exception:
            pass
        if is_block_response(e.code, e.headers, body_bytes):
            loc = ""
            try:
                loc = e.headers.get("Location") if e.headers else ""
            except Exception:
                pass
            raise BlockedUpstreamError(
                f"blocked HTTP {e.code}" + (f" -> {loc[:120]}" if loc else f": {e.reason}"),
                status_code=e.code,
            ) from e
        raise


def generate(prompt: str, model_id: int, think_mode: int, file_refs: list = None, extra_fields: dict = None) -> str:
    """Non-streaming generation with proxy rotation + retry."""
    body = _build_payload(prompt, model_id, think_mode, file_refs, extra_fields).encode()
    url = _get_url()
    headers = _build_headers()

    last_err = None
    attempts = _max_proxy_attempts()
    tried: set = set()

    for attempt in range(attempts):
        candidates = [c for c in POOL.candidates() if _client_key(c) not in tried]
        if not candidates:
            # Full cycle exhausted; allow re-try of best available.
            candidates = POOL.candidates()
            tried.clear()
        proxy = candidates[0] if candidates else None
        tried.add(_client_key(proxy))
        label = proxy or "direct"
        try:
            if HAS_HTTPX:
                text = _generate_once_httpx(url, body, headers, proxy)
            else:
                text = _generate_once_urllib(url, body, headers, proxy)
            POOL.mark_success(proxy)
            return text
        except BlockedUpstreamError as e:
            last_err = e
            POOL.mark_failure(proxy, str(e), force_rotate=True)
            log(f"Upstream blocked via {label}: {e}")
        except Exception as e:
            last_err = e
            reason = error_reason(e)
            POOL.mark_failure(proxy, reason, force_rotate=True)
            log(f"Retry {attempt+1}/{attempts} via {label}: {e}")
            if attempt < attempts - 1:
                time.sleep(CONFIG["retry_delay_sec"])
    raise last_err


def generate_stream(prompt: str, model_id: int, think_mode: int, file_refs: list = None, extra_fields: dict = None):
    """Streaming generation via httpx with proxy rotation on failure."""
    if not HAS_HTTPX:
        text = generate(prompt, model_id, think_mode, file_refs, extra_fields)
        if text:
            yield text
        return

    body = _build_payload(prompt, model_id, think_mode, file_refs, extra_fields)
    url = _get_url()
    headers = _build_headers()

    last_err = None
    attempts = _max_proxy_attempts()
    tried: set = set()
    emitted_raw_text = ""  # persists across retries so a resumed stream continues, not re-emits

    for attempt in range(attempts):
        candidates = [c for c in POOL.candidates() if _client_key(c) not in tried]
        if not candidates:
            candidates = POOL.candidates()
            tried.clear()
        proxy = candidates[0] if candidates else None
        tried.add(_client_key(proxy))
        label = proxy or "direct"
        client = _get_httpx_client(proxy)
        try:
            with client.stream("POST", url, content=body, headers=headers) as resp:
                # Read a small prefix to detect captcha/block before streaming out.
                # httpx stream: check status first.
                if is_block_response(resp.status_code, resp.headers, b""):
                    # consume body for better diagnostics
                    try:
                        peek = resp.read()
                    except Exception:
                        peek = b""
                    loc = resp.headers.get("Location") or resp.headers.get("location") or ""
                    raise BlockedUpstreamError(
                        f"blocked HTTP {resp.status_code}" + (f" -> {loc[:120]}" if loc else ""),
                        status_code=resp.status_code,
                    )
                if resp.status_code >= 400:
                    peek = resp.read()
                    if is_block_response(resp.status_code, resp.headers, peek):
                        raise BlockedUpstreamError(f"blocked HTTP {resp.status_code}", status_code=resp.status_code)
                    resp.raise_for_status()

                buf = ""
                for chunk in resp.iter_text():
                    buf += chunk
                    if "BardErrorInfo" in buf:
                        _raise_bard_error(buf)
                    # Early captcha body detection
                    if len(buf) < 500 and is_block_response(200, None, buf):
                        raise BlockedUpstreamError("blocked body (captcha/sorry)")
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        for t in _extract_texts_from_line(line):
                            if t == emitted_raw_text or emitted_raw_text.startswith(t):
                                continue
                            if not t.startswith(emitted_raw_text):
                                raise RuntimeError("Gemini stream content changed during retry")
                            delta = clean_text(t[len(emitted_raw_text):], strip=False)
                            emitted_raw_text = t
                            if delta:
                                yield delta
            POOL.mark_success(proxy)
            return
        except BlockedUpstreamError as e:
            last_err = e
            POOL.mark_failure(proxy, str(e), force_rotate=True)
            log(f"Stream blocked via {label}: {e}")
        except Exception as e:
            last_err = e
            reason = error_reason(e)
            if is_block_error(e):
                POOL.mark_failure(proxy, reason, force_rotate=True)
                log(f"Stream block-like via {label}: {reason}")
            else:
                POOL.mark_failure(proxy, reason, force_rotate=True)
                log(f"Stream retry {attempt+1}/{attempts} via {label}: {e}")
            if attempt < attempts - 1:
                time.sleep(CONFIG["retry_delay_sec"])
    raise last_err
