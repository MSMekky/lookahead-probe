"""Tiny HTTP helpers with retries; standard library only."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

UA = {"User-Agent": "lookahead-probe/0.1 (research)"}


def get_bytes(url: str, tries: int = 5, timeout: int = 60) -> bytes:
    err = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (400, 404):
                raise
            err = e
        except Exception as e:  # network hiccup
            err = e
        time.sleep(2 ** i)
    raise RuntimeError(f"GET failed after {tries} tries: {url}: {err}")


def get_json(url: str, **kw):
    return json.loads(get_bytes(url, **kw).decode("utf-8"))
