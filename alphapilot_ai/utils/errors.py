"""Error-handling helpers and the project's *fail-loud* policy.

WHY THIS MODULE EXISTS
----------------------
The codebase historically defaults to wrapping work in ``try / except
Exception`` blocks that ``logger.debug(...)`` or ``pass`` on failure.
The cost of that habit has been considerable — multiple bugs ran in
production for weeks before anyone noticed:

  - ``_find_similar_trades`` tuple-order mismatch → TypeError on every
    autonomous decision, swallowed by three nested excepts and surfaced
    only as a generic "Wallet tick error" in bot_engine.
  - ``close_trade`` referenced ``symbol`` / ``duration_minutes`` without
    binding them → NameError every call, swallowed → adaptive and
    autonomous learn hooks dead since the day they were wired.
  - Reflection HTTP timeouts → empty ActivityLog lines with no cause.

THE POLICY
----------
1. **Default to loud failure.** If you're catching an exception you do
   not specifically expect, prefer ``loud_exception(logger, message)``
   over ``logger.debug + pass``. It logs with stack trace at WARNING
   level and re-raises by default.

2. **Swallowed exceptions are explicit and labeled.** When you do want
   to swallow (best-effort metric writes, audit-log failures, etc.),
   use ``swallow_with_reason(logger, "...")`` so the swallow is visible
   in the source AND in the logs. No naked ``except Exception: pass``.

3. **Trust internal callers; validate boundary callers.** Only swallow
   exceptions at process boundaries (HTTP handlers that must return,
   scheduler ticks that must continue, persistence layers that must not
   block trades). Inside business logic, raise.

USAGE
-----
    from utils.errors import loud_exception, swallow_with_reason

    # Loud: re-raises after logging
    with loud_exception(logger, "couldn't compute mission inputs"):
        edge = compute_edge(signal, decision)

    # Loud, but pluggable retry on a specific recoverable exception
    with loud_exception(logger, "claude call failed", swallow=(httpx.ReadTimeout,)):
        result = claude_chat(prompt)

    # Explicit swallow at a process boundary (label required for grep-ability)
    with swallow_with_reason(logger, "ActivityLog write is best-effort"):
        s.add(ActivityLog(...))
"""
from __future__ import annotations

import contextlib
import logging
from typing import Iterable, Optional, Type, Union

# Type alias: a single Exception class or a tuple of them.
ExceptionTypes = Union[Type[BaseException], Iterable[Type[BaseException]]]


@contextlib.contextmanager
def loud_exception(
    logger: logging.Logger,
    message: str,
    *,
    swallow: Optional[ExceptionTypes] = None,
    level: int = logging.WARNING,
):
    """Loud-by-default exception logger.

    Logs the message + full traceback at ``level`` and re-raises. If
    ``swallow`` is provided, exceptions matching those types are logged
    at the same level and swallowed (returning normally). Everything
    else still propagates.

    The intent is to invert the codebase's default: a developer who
    wants to swallow has to opt in by listing the specific exception
    types they accept. Naked ``except Exception:`` becomes a code smell.

    Args:
        logger: The module logger (typically ``get_logger(__name__)``).
        message: Short human description of what was being attempted.
        swallow: Optional exception type or tuple of types to swallow
            instead of re-raising. Anything else still raises.
        level: Logging level (defaults to WARNING — visible by default).

    Example:
        # Raise on any failure, but log it loudly first
        with loud_exception(logger, "compute fingerprint"):
            fp = ctx.to_fingerprint()

        # Allow a known recoverable failure mode without dying
        with loud_exception(logger, "fetch price", swallow=(httpx.ReadTimeout,)):
            price = client.get_price(symbol)
    """
    try:
        yield
    except BaseException as exc:  # noqa: BLE001 — entire point is to catch broadly first
        if swallow is not None:
            allowed = (swallow,) if isinstance(swallow, type) else tuple(swallow)
            if isinstance(exc, allowed):
                logger.log(level, "[swallowed] %s: %s", message, exc, exc_info=True)
                return
        logger.log(level, "[loud] %s", message, exc_info=True)
        raise


@contextlib.contextmanager
def swallow_with_reason(
    logger: logging.Logger,
    reason: str,
    *,
    level: int = logging.DEBUG,
):
    """Explicit, labeled swallow for known-safe best-effort code paths.

    Use this ONLY when the surrounding code path is the safer default
    when work fails — e.g. writing an audit row whose absence is
    acceptable, sending a notification whose loss doesn't affect trading.

    The ``reason`` argument is required and is grep-able in source. It
    forces the swallow to be self-documenting: any future maintainer
    auditing for "why was this swallowed?" gets the answer at the call
    site rather than having to reconstruct intent from missing comments.

    Args:
        logger: The module logger.
        reason: Why swallowing here is OK. Required, never default.
        level: Log level for the swallowed exception (defaults to DEBUG
            because by definition we're not concerned about these).

    Example:
        # Audit-log write is fire-and-forget — trade must still close
        with swallow_with_reason(logger, "ActivityLog write is best-effort"):
            session.add(ActivityLog(...))

        # Heartbeat notification, never critical
        with swallow_with_reason(logger, "discord notify on tick is opportunistic"):
            notify_discord("tick complete")
    """
    if not reason:
        raise ValueError(
            "swallow_with_reason requires a non-empty reason. "
            "Use loud_exception() if you don't actually know why you're swallowing."
        )
    try:
        yield
    except BaseException as exc:  # noqa: BLE001
        logger.log(level, "[swallowed:%s] %s", reason, exc, exc_info=True)


def log_and_reraise(logger: logging.Logger, message: str, exc: BaseException) -> None:
    """Plain helper for the explicit ``except E as exc: log_and_reraise(...)``
    pattern when a context manager doesn't fit (e.g. inside an except clause
    that wants to log custom diagnostics before propagating).

    Always re-raises. There is no flag to suppress — by design.
    """
    logger.warning("[loud] %s: %s", message, exc, exc_info=True)
    raise exc
