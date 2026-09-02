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
