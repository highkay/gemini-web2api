"""Multimodal: Scotty resumable upload for Gemini image input."""
import re
import time
from typing import Optional

from .config import CONFIG
from .gemini import HAS_HTTPX, load_cookie, make_sapisidhash, _get_httpx_client, log
from .proxy_pool import POOL, is_block_response, error_reason


def _get_page_tokens() -> dict:
    """Fetch WIZ_global_data tokens from Gemini page (Push-ID, X-Client-Pctx)."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    cookie_str, sapisid = load_cookie()
    if cookie_str:
        headers["Cookie"] = cookie_str
    try:
        html = _http_get("https://gemini.google.com/app", headers=headers, timeout=30)
        tokens = {}
        for key, pattern in [
            ("push_id", r'"qKIAYe":"([^"]+)"'),
            ("pctx", r'"Ylro7b":"([^"]+)"'),
            ("at", r'"thykhd":"([^"]+)"'),
        ]:
            m = re.search(pattern, html)
            if m:
                tokens[key] = m.group(1)
        return tokens
    except Exception as e:
        log(f"Page token fetch failed: {e}")
        return {}


_page_tokens_cache = {"tokens": {}, "ts": 0}


def _cached_page_tokens() -> dict:
    now = time.time()
    if now - _page_tokens_cache["ts"] > 600:
        _page_tokens_cache["tokens"] = _get_page_tokens()
        _page_tokens_cache["ts"] = now
    return _page_tokens_cache["tokens"]


def _http_request(method: str, url: str, headers: dict, data: bytes = None, timeout: float = 30):
    """HTTP request via current proxy pool exit (httpx preferred for SOCKS)."""
    last_err = None
    for proxy in POOL.candidates():
        label = proxy or "direct"
        try:
            if not HAS_HTTPX:
                raise RuntimeError("httpx required for proxy pool multimodal path")
            client = _get_httpx_client(proxy)
            resp = client.request(method, url, content=data, headers=headers, timeout=timeout)
            if is_block_response(resp.status_code, resp.headers, resp.content):
                POOL.mark_failure(proxy, f"blocked HTTP {resp.status_code}", force_rotate=True)
                last_err = RuntimeError(f"blocked via {label}: HTTP {resp.status_code}")
                continue
            if resp.status_code >= 400:
                POOL.mark_failure(proxy, f"HTTP {resp.status_code}", force_rotate=True)
                last_err = RuntimeError(f"HTTP {resp.status_code} via {label}")
                continue
            POOL.mark_success(proxy)
            return resp
        except Exception as e:
            last_err = e
            POOL.mark_failure(proxy, error_reason(e), force_rotate=True)
            log(f"HTTP {method} via {label} failed: {e}")
    raise last_err or RuntimeError("all proxies failed")


def _http_get(url: str, headers: dict, timeout: float = 30) -> str:
    resp = _http_request("GET", url, headers=headers, timeout=timeout)
    return resp.content.decode("utf-8", errors="replace")


def upload_image(image_bytes: bytes, filename: str = "image.png", mime_type: str = "image/png") -> str:
    """Upload image via Scotty resumable upload. Returns file reference path."""
    tokens = _cached_page_tokens()
    push_id = tokens.get("push_id", "feeds/mcudyrk2a4khkz")
    pctx = tokens.get("pctx", "CgcSBWjK7pYx")

    cookie_str, sapisid = load_cookie()

    # Step 1: Initiate resumable upload
    start_headers = {
        "Push-ID": push_id,
        "X-Tenant-Id": "bard-storage",
        "X-Client-Pctx": pctx,
        "X-Goog-Upload-Header-Content-Length": str(len(image_bytes)),
        "X-Goog-Upload-Header-Content-Type": mime_type,
        "X-Goog-Upload-Protocol": "resumable",
        "X-Goog-Upload-Command": "start",
        "Content-Type": "application/x-www-form-urlencoded;charset=utf-8",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    if cookie_str:
        start_headers["Cookie"] = cookie_str
    if sapisid:
        start_headers["Authorization"] = make_sapisidhash(sapisid)

    start_url = "https://content-push.googleapis.com/upload/"
    resp = _http_request("POST", start_url, headers=start_headers, data=b"", timeout=30)

    upload_url = resp.headers.get("X-Goog-Upload-URL") or resp.headers.get("x-goog-upload-url")
    if not upload_url:
        raise RuntimeError(f"No upload URL in response headers: {dict(resp.headers)}")

    log(f"Upload session started: {upload_url[:80]}...")

    # Step 2: Upload file data + finalize
    upload_headers = {
        "X-Goog-Upload-Command": "upload, finalize",
        "X-Goog-Upload-Offset": "0",
        "Content-Type": "application/octet-stream",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    resp2 = _http_request("POST", upload_url, headers=upload_headers, data=image_bytes, timeout=60)

    file_ref = resp2.content.decode().strip()
    if not file_ref or not file_ref.startswith("/"):
        raise RuntimeError(f"Invalid file reference: {file_ref[:100]}")

    log(f"Image uploaded: {filename} -> {file_ref[:50]}...")
    return file_ref


def fetch_image_bytes(url: str) -> bytes:
    """Fetch image from URL (direct, no Gemini proxy needed)."""
    try:
        if HAS_HTTPX:
            import httpx
            with httpx.Client(timeout=30, follow_redirects=True) as client:
                r = client.get(url, headers={"User-Agent": "Mozilla/5.0"})
                r.raise_for_status()
                return r.content
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=30)
        return resp.read()
    except Exception as e:
        log(f"Image fetch failed: {e}")
        return b""
