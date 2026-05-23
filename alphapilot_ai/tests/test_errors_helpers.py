"""Tests for the loud-exception helpers in utils.errors.

These pin the *contract*: loud_exception re-raises by default, swallows
only the types you opt into; swallow_with_reason swallows but requires a
non-empty reason. If those defaults flip, the project's "fail-loud
policy" silently flips with them — exactly the failure mode this module
was meant to prevent.
"""
from __future__ import annotations

import logging

import pytest

from utils.errors import loud_exception, swallow_with_reason, log_and_reraise


def _make_logger() -> logging.Logger:
    """Logger that buffers messages so we can assert on what was logged."""
    log = logging.getLogger(f"test_errors_helpers.{id(object())}")
    log.handlers.clear()
    log.setLevel(logging.DEBUG)
    return log


def test_loud_exception_reraises_by_default():
    """The whole point of the helper: do not silently catch."""
    log = _make_logger()
    with pytest.raises(ValueError, match="boom"):
        with loud_exception(log, "while running thing"):
            raise ValueError("boom")


def test_loud_exception_swallows_only_listed_types():
    """When opted into, the matching type is swallowed; others propagate."""
    log = _make_logger()

    # Listed -> swallowed
    with loud_exception(log, "ok to lose", swallow=(KeyError,)):
        raise KeyError("missing")

    # Not listed -> still raises
    with pytest.raises(ValueError):
        with loud_exception(log, "not ok", swallow=(KeyError,)):
            raise ValueError("propagates")


def test_loud_exception_accepts_single_type_for_swallow():
    """Convenience: callers can pass a single class instead of a tuple."""
    log = _make_logger()
    with loud_exception(log, "single type", swallow=TimeoutError):
        raise TimeoutError("slow")


def test_loud_exception_logs_with_traceback():
    """Loud means visible. We require WARNING-level by default with exc_info."""
    log = _make_logger()
    records: list[logging.LogRecord] = []
    log.addHandler(logging.Handler())
    log.handlers[0].emit = records.append  # type: ignore[method-assign]
    log.setLevel(logging.DEBUG)

    with pytest.raises(RuntimeError):
        with loud_exception(log, "do thing"):
            raise RuntimeError("bad")

    assert records, "loud_exception did not log on failure"
    rec = records[0]
    assert rec.levelno == logging.WARNING
    # exc_info must be set so the formatter includes the traceback.
    assert rec.exc_info is not None
    assert "do thing" in rec.getMessage()


def test_swallow_with_reason_swallows():
    """Should NOT raise — the whole point is opt-in silence."""
    log = _make_logger()
    with swallow_with_reason(log, "audit log is best-effort"):
        raise IOError("disk gone")  # noqa: B904 — intentional


def test_swallow_with_reason_requires_non_empty_reason():
    """A naked swallow with no reason is exactly the failure mode the
    helper was designed to prevent. Empty reason -> ValueError so it
    surfaces during development."""
    log = _make_logger()
    with pytest.raises(ValueError, match="non-empty reason"):
        with swallow_with_reason(log, ""):
            pass


def test_swallow_with_reason_logs_reason_for_grep():
    """The reason must appear in the log message so an operator scanning
    `[swallowed:` lines can tell what was lost."""
    log = _make_logger()
    records: list[logging.LogRecord] = []
    log.addHandler(logging.Handler())
    log.handlers[0].emit = records.append  # type: ignore[method-assign]

    with swallow_with_reason(log, "audit log is best-effort"):
        raise ValueError("dummy")

    assert records, "swallow did not log"
    msg = records[0].getMessage()
    assert "swallowed:audit log is best-effort" in msg, (
        f"reason label missing from swallow log line: {msg!r}"
    )


def test_log_and_reraise_always_reraises():
    """Plain helper for use inside an explicit except. No flag to suppress."""
    log = _make_logger()
    with pytest.raises(KeyError):
        try:
            raise KeyError("missing")
        except KeyError as exc:
            log_and_reraise(log, "looking up key", exc)
