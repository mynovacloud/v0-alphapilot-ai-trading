"""Regression: the universe builder must not feed the bot micro-cap junk.

The old builder claimed to rank by 24h volume but actually appended every
non-priority symbol ALPHABETICALLY. With limit=200 that pulled in
`00-USD`, `A8-USD`, `BOBBOB-USD`, `DOOD-USD` — thin micro-caps the bot
then traded and lost on (fees + slippage swamp any edge there).

`coinbase_usd_universe` now restricts to the curated `_LIQUID_UNIVERSE`
by default. These tests use the module cache to avoid a network call.
"""
from __future__ import annotations

import time

from connectors.universe import coinbase_usd_universe, _CACHE, _LIQUID_UNIVERSE


def _product(pid: str) -> dict:
    return {"product_id": pid, "base": pid.split("-")[0], "quote": "USD",
            "price": 0.0, "volume_24h": 0.0}


def _seed(*product_ids: str) -> None:
    """Prime the universe cache so coinbase_usd_universe skips the network."""
    _CACHE["coinbase_usd"] = ([_product(p) for p in product_ids], time.time())


_MIXED = ("BTC-USD", "ETH-USD", "SOL-USD", "00-USD", "A8-USD",
          "BOBBOB-USD", "DOOD-USD", "FUN1-USD")


def test_liquid_only_drops_microcap_junk():
    _seed(*_MIXED)
    ids = {r["product_id"] for r in coinbase_usd_universe(limit=200)}
    assert {"BTC-USD", "ETH-USD", "SOL-USD"} <= ids
    for junk in ("00-USD", "A8-USD", "BOBBOB-USD", "DOOD-USD", "FUN1-USD"):
        assert junk not in ids, f"{junk} should have been filtered out"


def test_liquid_only_false_keeps_everything():
    _seed(*_MIXED)
    ids = {r["product_id"] for r in coinbase_usd_universe(limit=200, liquid_only=False)}
    assert "BOBBOB-USD" in ids and "00-USD" in ids


def test_priority_ordering_is_honored():
    # Seeded out of order; BTC must still come before SOL (its index in
    # _LIQUID_UNIVERSE is lower).
    _seed("SOL-USD", "TIA-USD", "BTC-USD", "ETH-USD")
    uni = coinbase_usd_universe(limit=200)
    order = [r["product_id"] for r in uni]
    assert order.index("BTC-USD") < order.index("ETH-USD") < order.index("SOL-USD")


def test_limit_caps_result():
    _seed(*_LIQUID_UNIVERSE)
    assert len(coinbase_usd_universe(limit=10)) == 10


def test_curated_list_is_sane():
    # No duplicates, all USD pairs — a typo here silently shrinks the universe.
    assert len(_LIQUID_UNIVERSE) == len(set(_LIQUID_UNIVERSE))
    assert all(s.endswith("-USD") for s in _LIQUID_UNIVERSE)
    assert len(_LIQUID_UNIVERSE) >= 40  # enough breadth to diversify 25 slots
