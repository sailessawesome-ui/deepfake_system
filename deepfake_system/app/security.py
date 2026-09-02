"""Transport and browser-side hardening — IR 3.4.1, security requirement 2.

"The data sent between the user interface (e.g. the browser extension) and
the backend processing server should be heavily encrypted to avoid
interception or manipulation, particularly because users have sensitive
topics (such as financial fraud and explicit content) that they are
worried about."

Encryption in transit is TLS, and TLS is terminated at the reverse proxy
(`deploy/nginx.conf`) rather than in Python — uvicorn can serve TLS, but a
proxy is where certificate renewal, OCSP stapling and HTTP/2 actually
live. What this module adds is the half a proxy cannot do for you:

- **HSTS**, so a browser that has seen the site once refuses to downgrade
  to http even if a user types it. Emitted only over a secure connection —
  sending it over http is meaningless, and sending it from localhost
  poisons the developer's own browser for every other localhost project.
- **A strict Content-Security-Policy.** The interface has no inline
  `<script>` blocks at all, which is unusual and worth exploiting: script
  execution is limited to same-origin files with no `unsafe-inline` and no
  `unsafe-eval`, so an injected `<script>` tag simply does not run. That
  is the control that would have contained the stored-XSS hole in the
  account badge rather than relying on escaping alone.
- **Redirect to https**, when told it is behind a TLS-terminating proxy.

`X-Forwarded-Proto` is trusted here, which is only safe because the app is
meant to sit behind a proxy that overwrites it. Exposed directly to the
internet, a client could set that header itself and claim to be secure —
so the redirect is opt-in via DFD_FORCE_HTTPS rather than on by default.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from starlette.middleware.base import BaseHTTPMiddleware  # noqa: E402
from starlette.responses import RedirectResponse  # noqa: E402

from config import _env, _env_bool  # noqa: E402

# One year, the value HSTS preload lists require.
HSTS_SECONDS = int(_env("DFD_HSTS_SECONDS", default="31536000"))
FORCE_HTTPS = _env_bool("DFD_FORCE_HTTPS", default=False)

# style-src needs 'unsafe-inline' because the page and the report sheet
# use a handful of style="" attributes. script-src deliberately does not:
# there is no inline script anywhere in the interface, so the strong form
# costs nothing. Do not add 'unsafe-inline' to script-src to fix a bug —
# move the script into app.js instead.
CSP = "; ".join([
    "default-src 'self'",
    "script-src 'self'",
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
    "font-src 'self' https://fonts.gstatic.com",
    # data: is required: face crops and Grad-CAM overlays are returned as
    # base64 JPEGs precisely so they never touch disk (zero retention).
    "img-src 'self' data:",
    "connect-src 'self'",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    "frame-ancestors 'none'",
])

HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    # The tool needs none of these. Denying them means a compromised
    # script cannot quietly reach for the webcam or microphone of an
    # analyst reviewing sensitive material.
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), "
                          "payment=(), usb=(), interest-cohort=()",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
}


def is_secure(request) -> bool:
    """True when the browser's connection is TLS.

    Behind a proxy the ASGI scheme is http even when the client used
    https, so the forwarded header is what actually carries the answer.
    """
    proto = request.headers.get("x-forwarded-proto", "").split(",")[0].strip()
    return proto == "https" or request.url.scheme == "https"


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        secure = is_secure(request)

        if FORCE_HTTPS and not secure and request.method in ("GET", "HEAD"):

            url = request.url.replace(scheme="https")
            return RedirectResponse(str(url), status_code=307)

        response = await call_next(request)

        for key, value in HEADERS.items():
            response.headers.setdefault(key, value)
        response.headers.setdefault("Content-Security-Policy", CSP)

        if secure:
            response.headers.setdefault(
                "Strict-Transport-Security",
                f"max-age={HSTS_SECONDS}; includeSubDomains")


        if request.url.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control",
                                        "no-store, no-cache, must-revalidate")
            response.headers.setdefault("Pragma", "no-cache")
        return response


def status() -> dict:
    """What the transport layer is actually enforcing, for /api/status."""
    return {
        "force_https": FORCE_HTTPS,
        "hsts_seconds": HSTS_SECONDS,
        "csp": True,
        "cookie_secure": _env_bool("DFD_COOKIE_SECURE", "DF_COOKIE_SECURE",
                                   default=bool(os.environ.get(
                                       "RAILWAY_ENVIRONMENT"))),
    }
