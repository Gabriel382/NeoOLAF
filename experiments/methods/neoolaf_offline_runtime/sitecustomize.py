"""Optional runtime policy that disables Wikipedia HTTP requests.

Python imports ``sitecustomize`` automatically when this directory is included
in ``PYTHONPATH``. This keeps the NeoOLAF source tree unchanged while allowing
the benchmark notebook to run in a reproducible, Wikipedia-free mode.
"""

from __future__ import annotations

import json
import os
from urllib.parse import urlparse


def _wikipedia_is_disabled() -> bool:
    """Return whether the benchmark requested offline Wikipedia behavior."""

    value = os.environ.get("NEOOLAF_DISABLE_WIKIPEDIA", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _is_wikipedia_url(url: object) -> bool:
    """Identify Wikipedia and Wikimedia API hosts without blocking other HTTP."""

    try:
        hostname = (urlparse(str(url)).hostname or "").lower()
    except Exception:
        return False
    return (
        hostname == "wikipedia.org"
        or hostname.endswith(".wikipedia.org")
        or hostname == "wikimedia.org"
        or hostname.endswith(".wikimedia.org")
    )


if _wikipedia_is_disabled():
    try:
        import requests
        from requests import Response
    except Exception:
        requests = None

    if requests is not None:
        _original_session_request = requests.sessions.Session.request

        def _offline_session_request(self, method, url, *args, **kwargs):
            """Return an empty successful MediaWiki result for blocked hosts."""

            if not _is_wikipedia_url(url):
                return _original_session_request(self, method, url, *args, **kwargs)

            payload = {
                "batchcomplete": "",
                "continue": {},
                "query": {
                    "search": [],
                    "pages": {},
                },
            }
            response = Response()
            response.status_code = 200
            response.url = str(url)
            response.reason = "Wikipedia disabled by NeoOLAF benchmark policy"
            response.headers["Content-Type"] = "application/json; charset=utf-8"
            response._content = json.dumps(payload).encode("utf-8")
            response.encoding = "utf-8"
            return response

        requests.sessions.Session.request = _offline_session_request
        print("[NeoOLAF benchmark] Wikipedia lookups disabled by runtime policy.")
