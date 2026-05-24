"""Phase A++: the focused-universe Settings field.

The operator can paste any list of symbols into Settings; the bot
reads it on every tick via get_active_universe() and trades exactly
those. Empty/invalid input falls back to the hardcoded 8-major
default. These tests cover the parser, the resolver, and the live
universe-builder integration.
"""
from __future__ import annotations

import time

from connectors.universe import (
    _BROAD_UNIVERSE,
    _FOCUSED_UNIVERSE,
    _LIQUID_UNIVERSE,
    _CACHE,
    coinbase_usd_universe,
    get_active_universe,
    parse_symbols,
)
from config import bot_config as _bc


# --------------------------------------------------------------------------
# parse_symbols — the input layer
# --------------------------------------------------------------------------

def test_parse_basic_comma_separated():
    assert parse_symbols("BTC-USD, ETH-USD, SOL-USD") == ("BTC-USD", "ETH-USD", "SOL-USD")


def test_parse_handles_whitespace_and_newlines():
    assert parse_symbols("BTC-USD\nETH-USD  SOL-USD\tLINK-USD") == (
        "BTC-USD", "ETH-USD", "SOL-USD", "LINK-USD",
    )


def test_parse_auto_suffixes_usd_when_missing():
    """Bare tickers without -USD get suffixed automatically — friendlier
    paste of e.g. a copy-pasted symbol list."""
    assert parse_symbols("BTC, ETH, sol") == ("BTC-USD", "ETH-USD", "SOL-USD")


def test_parse_strips_dollar_prefix():
    assert parse_symbols("$BTC $ETH") == ("BTC-USD", "ETH-USD")


def test_parse_dedupes_preserving_order():
    assert parse_symbols("BTC, ETH, BTC, sol, ETH-USD") == ("BTC-USD", "ETH-USD", "SOL-USD")


def test_parse_drops_invalid_shapes_silently():
    """Garbage tokens don't crash and don't pollute the output —
    they're silently dropped."""
    out = parse_symbols("BTC, totally-not-a-symbol, ETH, --, !!, SOL")
    assert "BTC-USD" in out and "ETH-USD" in out and "SOL-USD" in out
    # 'totally-not-a-symbol' has 'totally' base which is alnum and ends -symbol
    # so doesn't pass — let's be specific about what passes.
    assert all(s.endswith("-USD") for s in out)


def test_parse_empty_string_returns_empty_tuple():
    assert parse_symbols("") == ()
    assert parse_symbols("   ") == ()


# --------------------------------------------------------------------------
# get_active_universe — the resolver
# --------------------------------------------------------------------------

def test_default_active_is_the_focused_8():
    """With no Settings override, active = the hardcoded focused list."""
    _bc.set_many({"bot_focused_symbols": ""})
    assert get_active_universe() == _FOCUSED_UNIVERSE
    assert len(_FOCUSED_UNIVERSE) == 8


def test_setting_override_takes_effect():
    """A non-empty Settings value REPLACES the default — no merging."""
    _bc.set_many({"bot_focused_symbols": "BTC-USD, ETH-USD, DOGE-USD"})
    try:
        assert get_active_universe() == ("BTC-USD", "ETH-USD", "DOGE-USD")
    finally:
        _bc.set_many({"bot_focused_symbols": ""})


def test_garbage_setting_falls_back_to_default():
    """If the operator saves all-invalid input, the resolver doesn't
    crash and doesn't leave them with zero trading symbols — it falls
    back to the hardcoded default."""
    _bc.set_many({"bot_focused_symbols": "not-real, --, !!"})
    try:
        assert get_active_universe() == _FOCUSED_UNIVERSE
    finally:
        _bc.set_many({"bot_focused_symbols": ""})


def test_setting_supports_bare_ticker_input():
    """A casual paste of bare tickers (no -USD) gets cleaned up."""
    _bc.set_many({"bot_focused_symbols": "BTC ETH SHIB PEPE"})
    try:
        assert get_active_universe() == ("BTC-USD", "ETH-USD", "SHIB-USD", "PEPE-USD")
    finally:
        _bc.set_many({"bot_focused_symbols": ""})


# --------------------------------------------------------------------------
# coinbase_usd_universe — the live builder honors the resolver
# --------------------------------------------------------------------------

def _seed_cache(*product_ids: str) -> None:
    """Bypass the network — preload the cache with synthetic products
    so we can verify filtering without a real Coinbase fetch."""
    rows = [{"product_id": p, "base": p.split("-")[0], "quote": "USD",
             "price": 0.0, "volume_24h": 0.0} for p in product_ids]
    _CACHE["coinbase_usd"] = (rows, time.time())


def test_coinbase_universe_reflects_settings_override_immediately():
    """The user's whole concern: 'when I save Settings, does the bot
    actually trade those?' Verify the end-to-end path: change setting,
    call the builder, the result reflects the change without restart."""
    _seed_cache("BTC-USD", "ETH-USD", "SOL-USD", "PEPE-USD", "ABC-USD", "XYZ-USD")

    _bc.set_many({"bot_focused_symbols": "BTC, ETH, PEPE"})
    try:
        ids = {r["product_id"] for r in coinbase_usd_universe(limit=100)}
        assert ids == {"BTC-USD", "ETH-USD", "PEPE-USD"}
    finally:
        _bc.set_many({"bot_focused_symbols": ""})

    # Empty setting -> back to the hardcoded default; the seeded BTC/ETH/SOL
    # all belong to _FOCUSED_UNIVERSE so they stay, the seeded ABC/XYZ drop.
    ids = {r["product_id"] for r in coinbase_usd_universe(limit=100)}
    assert "BTC-USD" in ids and "ETH-USD" in ids and "SOL-USD" in ids
    assert "ABC-USD" not in ids


def test_coinbase_universe_respects_operator_ordering():
    """The order the operator typed should be the order the bot
    evaluates — first symbols get scored first within the tick budget."""
    _seed_cache("BTC-USD", "ETH-USD", "SOL-USD", "AAVE-USD")
    _bc.set_many({"bot_focused_symbols": "AAVE, SOL, ETH, BTC"})
    try:
        order = [r["product_id"] for r in coinbase_usd_universe(limit=100)]
        assert order == ["AAVE-USD", "SOL-USD", "ETH-USD", "BTC-USD"]
    finally:
        _bc.set_many({"bot_focused_symbols": ""})


def test_broad_universe_constant_is_preserved():
    """The fallback library of 50+ symbols must stay reachable for any
    callers that explicitly want the wide view (liquid_only=False)."""
    assert len(_BROAD_UNIVERSE) >= 40
    assert "BTC-USD" in _BROAD_UNIVERSE


def test_liquid_universe_alias_is_backward_compatible():
    """Older tests and the harness import _LIQUID_UNIVERSE. It must
    still resolve to the focused default."""
    assert _LIQUID_UNIVERSE == _FOCUSED_UNIVERSE
