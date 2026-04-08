from core_lib.utils.url_utils import normalize_url_with_scheme


def test_normalize_url_adds_https_when_missing_scheme() -> None:
    assert normalize_url_with_scheme("www.example.com") == "https://www.example.com"
    assert normalize_url_with_scheme("example.com/path") == "https://example.com/path"


def test_normalize_url_keeps_existing_scheme() -> None:
    assert normalize_url_with_scheme("https://example.com") == "https://example.com"
    assert normalize_url_with_scheme("http://example.com") == "http://example.com"


def test_normalize_url_handles_protocol_relative() -> None:
    assert normalize_url_with_scheme("//example.com/path") == "https://example.com/path"


def test_normalize_url_handles_empty_input() -> None:
    assert normalize_url_with_scheme("") == ""
    assert normalize_url_with_scheme("   ") == ""
