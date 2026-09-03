"""TLS helpers for Hugging Face downloads behind corporate proxies."""

from __future__ import annotations

import os
import ssl


def insecure_ssl_requested() -> bool:
    flag = os.environ.get("HTM_INSECURE_SSL", "").strip().lower()
    return flag in {"1", "true", "yes", "on"}


def apply_insecure_ssl() -> None:
    """Disable TLS verification for this process.

    Needed when a local proxy inspects HTTPS and the Python runtime
    does not trust the interceptor certificate.
    """
    ssl._create_default_https_context = ssl._create_unverified_context  # noqa: SLF001
    os.environ.setdefault("PYTHONHTTPSVERIFY", "0")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    try:
        import httpx

        original = httpx.Client.__init__

        def _init(self, *args, **kwargs):
            kwargs["verify"] = False
            return original(self, *args, **kwargs)

        httpx.Client.__init__ = _init  # type: ignore[method-assign]
    except Exception:
        pass


def ensure_ssl(insecure: bool | None) -> None:
    if insecure is True or (insecure is None and insecure_ssl_requested()):
        apply_insecure_ssl()
