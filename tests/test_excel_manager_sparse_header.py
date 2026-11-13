"""Prevent per-column XML reparsing without dropping distant real content."""
from openpyxl import Workbook
from openpyxl.styles import PatternFill

from core_lib.tools.excel_manager import ExcelManager
from core_lib.utils.language_utils import LanguageUtils


def test_sparse_header_is_read_in_one_pass_and_far_columns_are_retained(tmp_path, monkeypatch):
    wb = Workbook()
    ws = wb.active
    ws.append(["ID", "Question", "Answer"])
    ws.append(["Q1", "Describe the control.", None])
    ws["ZZ1"] = "Distant requirement"
    ws["ZZ2"] = "Preserve intermediate evidence."
    ws["XFC1"] = "Far requirement"
    ws["XFC2"] = "Preserve last-column evidence."
    ws["XFD32"].fill = PatternFill("solid", fgColor="FFFFFF")
    path = tmp_path / "sparse.xlsx"
    wb.save(path)
    monkeypatch.setattr(LanguageUtils, "detect_language", lambda *_a, **_kw: "en")
    manager = ExcelManager(str(path))
    manager.load()
    sheet = manager.wb.active
    original_cell = sheet.cell
    calls = []

    def counted_cell(*args, **kwargs):
        calls.append((args, kwargs))
        return original_cell(*args, **kwargs)

    monkeypatch.setattr(sheet, "cell", counted_cell)
    try:
        block = manager.to_json_ir()["document"]["sheets"][0]["blocks"][0]
    finally:
        manager.close()
    # Distant-column sampling may access at most six columns at three rows.
    # Before the fix the header adds another 16,384 XML parses here.
    assert len(calls) <= 18
    assert block["columns"]["ZZ"]["header"] == "Distant requirement"
    assert block["columns"]["XFC"]["header"] == "Far requirement"
    assert block["rows"][0]["cells"]["ZZ"] == "Preserve intermediate evidence."
    assert block["rows"][0]["cells"]["XFC"] == "Preserve last-column evidence."
