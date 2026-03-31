import os
import io
import logging
from pathlib import Path

from core_lib import setup_logging, get_module_logger, get_last_logging_config
from core_lib.tracing.logger import ColorizedConsoleFormatter


class FakeTerminalStream(io.StringIO):
    def __init__(self, *, is_tty: bool):
        super().__init__()
        self._is_tty = is_tty

    def isatty(self):
        return self._is_tty

    def reconfigure(self, **kwargs):
        return None

def test_setup_logging_basic(tmp_path: Path):
    log_file = tmp_path / "test.log"
    logger = setup_logging(level="DEBUG", file_logging=True, file_path=str(log_file), force=True)
    logger.debug("debug message")
    logger.info("info message")

    # Ensure config captured
    cfg = get_last_logging_config()
    assert cfg["file_logging"] is True
    assert cfg["file_path"].endswith("test.log")

    # Flush handlers
    for h in logging.getLogger().handlers:
        try:
            h.flush()
        except Exception:
            pass

    assert log_file.exists(), "Log file should be created when file_logging enabled"
    content = log_file.read_text(encoding="utf-8")
    assert "info message" in content


def test_get_module_logger_does_not_initialize_again():
    # Calling get_module_logger should not reconfigure handlers
    before = len(logging.getLogger().handlers)
    _ = get_module_logger()
    after = len(logging.getLogger().handlers)
    assert after == before


def test_colorized_console_formatter_uses_different_colors_for_warning_and_error():
    formatter = ColorizedConsoleFormatter("%(levelname)s:%(message)s")

    warning_record = logging.LogRecord("test", logging.WARNING, __file__, 10, "watch out", (), None)
    error_record = logging.LogRecord("test", logging.ERROR, __file__, 11, "boom", (), None)

    warning_output = formatter.format(warning_record)
    error_output = formatter.format(error_record)

    assert warning_output.startswith("\x1b[33;1mWARNING:watch out")
    assert error_output.startswith("\x1b[31;1mERROR:boom")
    assert warning_output.endswith("\x1b[0m")
    assert error_output.endswith("\x1b[0m")


def test_setup_logging_uses_color_formatter_for_tty_console(monkeypatch):
    from core_lib.tracing import logger as logger_module

    fake_stream = FakeTerminalStream(is_tty=True)
    monkeypatch.setattr(logger_module.sys, "stdout", fake_stream)

    logger = setup_logging(level="INFO", force=True)
    logger.warning("warning message")
    logger.error("error message")

    for handler in logging.getLogger().handlers:
        try:
            handler.flush()
        except Exception:
            pass

    output = fake_stream.getvalue()
    assert isinstance(logging.getLogger().handlers[0].formatter, ColorizedConsoleFormatter)
    assert "\x1b[33;1m" in output
    assert "\x1b[31;1m" in output


def test_setup_logging_skips_color_formatter_for_non_tty_console(monkeypatch):
    from core_lib.tracing import logger as logger_module

    fake_stream = FakeTerminalStream(is_tty=False)
    monkeypatch.setattr(logger_module.sys, "stdout", fake_stream)

    logger = setup_logging(level="INFO", force=True)
    logger.warning("plain warning")

    for handler in logging.getLogger().handlers:
        try:
            handler.flush()
        except Exception:
            pass

    output = fake_stream.getvalue()
    assert not isinstance(logging.getLogger().handlers[0].formatter, ColorizedConsoleFormatter)
    assert "\x1b[" not in output
