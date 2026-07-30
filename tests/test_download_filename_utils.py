"""Unit tests for build_safe_download_filename and sanitize_filename_component in core_lib.utils.file_utils."""
import pytest
from core_lib.utils.file_utils import sanitize_filename_component, build_safe_download_filename


def test_sanitize_filename_component_basic():
    assert sanitize_filename_component("Simple Title") == "Simple_Title"
    assert sanitize_filename_component("Title / With \\ Slashes") == "Title_With_Slashes"
    assert sanitize_filename_component('Title : "With" <Invalid> ? * | Chars') == "Title_With_Invalid_Chars"


def test_sanitize_filename_component_windows_reserved():
    assert sanitize_filename_component("CON") == "CON_doc"
    assert sanitize_filename_component("aux") == "aux_doc"
    assert sanitize_filename_component("COM1") == "COM1_doc"
    assert sanitize_filename_component("NUL") == "NUL_doc"


def test_sanitize_filename_component_truncation():
    long_name = "A" * 200
    sanitized = sanitize_filename_component(long_name, max_length=50)
    assert len(sanitized) == 50
    assert sanitized == "A" * 50


def test_build_safe_download_filename_with_title_and_version():
    filename = build_safe_download_filename(
        title="Security Questionnaire 2026",
        version=1,
        extension_or_filename="file_1234.xlsx"
    )
    assert filename == "Security_Questionnaire_2026_v1.xlsx"


def test_build_safe_download_filename_version_string():
    filename = build_safe_download_filename(
        title="RFP Technical Assessment",
        version="v2.0",
        extension_or_filename="path/to/doc.docx"
    )
    assert filename == "RFP_Technical_Assessment_v2.0.docx"


def test_build_safe_download_filename_prefix():
    filename = build_safe_download_filename(
        title="Vendor Assessment",
        version=3,
        extension_or_filename="test.xlsx",
        prefix="answered"
    )
    assert filename == "answered_Vendor_Assessment_v3.xlsx"


def test_build_safe_download_filename_invalid_chars_in_title():
    filename = build_safe_download_filename(
        title="Vendor Q&A: Security / Compliance ? 2026",
        version=1,
        extension_or_filename="data.xlsx"
    )
    assert filename == "Vendor_Q&A_Security_Compliance_2026_v1.xlsx"


def test_build_safe_download_filename_fallback_title():
    filename = build_safe_download_filename(
        title="",
        version=2,
        extension_or_filename="original_internal_name.pdf"
    )
    assert filename == "original_internal_name_v2.pdf"
