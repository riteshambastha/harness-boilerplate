"""Example tool: fetch a web page as plain text.

Shows the shape of a tool that reaches the outside world: validate the input,
set a timeout, and cap the output so one page can't flood the context window.
Uses stdlib urllib; `truststore` verifies TLS with the OS certificate store
(avoids the common CERTIFICATE_VERIFY_FAILED error on macOS Python installs).
"""

from __future__ import annotations

import html
import re
import ssl
import urllib.request
from typing import Annotated

import truststore
from pydantic import Field

from harness.tools import tool


def _html_to_text(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", raw)
    text = html.unescape(re.sub(r"(?s)<[^>]+>", " ", raw))
    return re.sub(r"\s+", " ", text).strip()


@tool
def web_fetch(
    url: Annotated[str, Field(description="An http(s) URL to fetch")],
    max_chars: Annotated[int, Field(description="Truncate the text to this many characters", le=20_000)] = 4000,
) -> str:
    """Fetch a web page and return its visible text (HTML tags stripped)."""
    if not url.startswith(("http://", "https://")):
        raise ValueError("only http:// and https:// URLs are allowed")
    request = urllib.request.Request(url, headers={"User-Agent": "harness-starter/0.1"})
    tls = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    with urllib.request.urlopen(request, timeout=10, context=tls) as resp:
        body = resp.read(2_000_000).decode(resp.headers.get_content_charset() or "utf-8", "replace")
    text = _html_to_text(body) if "<" in body else body
    return text[:max_chars] + (" …[truncated]" if len(text) > max_chars else "")
