"""Run the app over HTTPS locally, with a self-signed certificate.

    python scripts/dev_tls.py            # https://127.0.0.1:8443

Production TLS is terminated by nginx (`deploy/nginx.conf`); this exists so
the encrypted path can be demonstrated and tested without a domain or a
real certificate — including in a viva, on a laptop, offline.

The browser will warn that the certificate is untrusted. That is correct
behaviour and worth saying out loud rather than clicking past: the
certificate proves nothing about identity because nothing signed it. It
encrypts the connection, which is what is being demonstrated; it does not
authenticate the server, which is what a real CA adds.

Certificates are written to `runs/devcert/` and gitignored with the rest
of `runs/`.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CERT_DIR = ROOT / "runs" / "devcert"
CERT = CERT_DIR / "dev-cert.pem"
KEY = CERT_DIR / "dev-key.pem"

# SAN is required: browsers have ignored the legacy CommonName field for
# years, and a certificate without it fails before the warning page.
_CONF = """[req]
distinguished_name = dn
x509_extensions = v3
prompt = no
[dn]
CN = localhost
O = Deepfake Forensics (development)
[v3]
subjectAltName = DNS:localhost, DNS:127.0.0.1, IP:127.0.0.1, IP:::1
basicConstraints = critical, CA:FALSE
keyUsage = critical, digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth
"""


def ensure_cert() -> bool:
    if CERT.exists() and KEY.exists():
        print(f"[tls] reusing {CERT}")
        return True

    openssl = shutil.which("openssl")
    if not openssl:
        print("[tls] openssl not found on PATH.\n"
              "     Git for Windows ships one at "
              "C:\\Program Files\\Git\\usr\\bin\\openssl.exe,\n"
              "     or install it and re-run.")
        return False

    CERT_DIR.mkdir(parents=True, exist_ok=True)
    conf = CERT_DIR / "openssl.cnf"
    conf.write_text(_CONF, encoding="utf8")

    print("[tls] generating a self-signed certificate for localhost ...")
    proc = subprocess.run(
        [openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes",
         "-keyout", str(KEY), "-out", str(CERT),
         "-days", "365", "-config", str(conf)],
        capture_output=True, text=True)
    if proc.returncode != 0:
        print("[tls] openssl failed:\n" + (proc.stderr or "")[:800])
        return False

    try:
        os.chmod(KEY, 0o600)          # no-op on Windows, correct on Linux
    except OSError:
        pass
    print(f"[tls] wrote {CERT}")
    return True


def main() -> int:
    if not ensure_cert():
        return 2

    port = int(os.environ.get("PORT", "8443"))
    # Tell the app it is on a secure connection so it emits HSTS and marks
    # the session cookie Secure — the same code path production takes.
    os.environ.setdefault("DFD_COOKIE_SECURE", "1")

    try:
        import uvicorn
    except ImportError:
        print("uvicorn is not installed.  pip install -r requirements-web.txt")
        return 2

    print(f"\n  https://127.0.0.1:{port}\n"
          "  The certificate is self-signed, so the browser will warn once.\n")
    uvicorn.run("app.server:app", host="127.0.0.1", port=port,
                ssl_certfile=str(CERT), ssl_keyfile=str(KEY), reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
