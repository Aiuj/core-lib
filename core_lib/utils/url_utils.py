"""URL utility helpers for common URL normalization tasks."""

from urllib.parse import urlparse


def normalize_url_with_scheme(url: str) -> str:
    """Normalize a URL by ensuring it has an explicit scheme.

    If no scheme is provided, defaults to ``https://``.
    """
    normalized = (url or "").strip()
    if not normalized:
        return normalized

    parsed = urlparse(normalized)
    if parsed.scheme:
        return normalized

    if normalized.startswith("//"):
        return f"https:{normalized}"

    return f"https://{normalized}"
