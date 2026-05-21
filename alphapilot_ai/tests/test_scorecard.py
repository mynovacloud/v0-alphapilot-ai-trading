"""Trade-quality scorecard endpoint.

Pins the aggregation contract:
  - decision-source counts come from ClaudeDecision.source/.action
  - trade-calibration counts come from PaperTrade.calibration_source/_sample_size
  - reflection dedup numbers come from ActivityLog "Reflection saved..." rows
  - top_patterns come from the autonomous engine's _patterns table

Endpoint must:
  1. Not crash on an empty DB.
  2. Return a stable response shape (totals + per-category breakdowns).
  3. Honor the session cutoff (rows older than cutoff are excluded).
  4. Be safe to poll on the feed cadence — no expensive queries.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from database.db import session_scope
from database.models import (
    ActivityLog,
    ClaudeDecision,
    PaperTrade,
    Wallet,
)


def _call_endpoint() -> dict:
    """Invoke training_scorecard() as a function and decode its JSON body."""
    from app.web import training_scorecard
    resp = training_scorecard()
    return json.loads(resp.body.decode("utf-8"))


def _wipe_db():
    """Clean the test DB between cases — each test runs in isolation."""
    with session_scope() as s:
        s.query(ActivityLog).delete()
        s.query(ClaudeDecision).delete()
        s.query(PaperTrade).delete()
        s.query(Wallet).delete()


@pytest.fixture(autouse=True)
def _clean_db():
    _wipe_db()
    yield
    _wipe_db()


def test_empty_db_returns_zero_totals():
    """Smoke test: no wallets, no decisions, no trades → endpoint still
    returns a sane shape with zeros."""
    data = _call_endpoint()
    assert data["ok"] is True
    assert data["decisions"]["total"] == 0
    assert data["trades"]["total"] == 0
    assert data["reflections"]["total"] == 0
    assert data["reflections"]["lessons_new"] == 0
    assert data["reflections"]["lessons_reinforced"] == 0
    assert data["reflections"]["dedup_ratio"] == 0.0
    assert isinstance(data["top_patterns"], list)


def test_decisions_grouped_by_source_and_action():
    """When ClaudeDecision rows exist, the endpoint must group them by
    (source, action) and report counts per source. This is the most
    important UI block — it tells the operator "what's making my
    decisions today?"."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    with session_scope() as s:
        w = Wallet(name="test", platform="paper",
                   bankroll_reset_at=now - timedelta(hours=2))
        s.add(w)
        s.flush()
        # Three autonomous-source decisions, mixed actions.
        s.add(ClaudeDecision(wallet_id=w.id, symbol="BTC-USD",
                             source="autonomous", action="HOLD", created_at=now))
        s.add(ClaudeDecision(wallet_id=w.id, symbol="ETH-USD",
                             source="autonomous", action="HOLD", created_at=now))
        s.add(ClaudeDecision(wallet_id=w.id, symbol="SOL-USD",
                             source="autonomous", action="BUY", created_at=now))
        # Two training_passthrough decisions, both BUY.
        s.add(ClaudeDecision(wallet_id=w.id, symbol="DOGE-USD",
                             source="training_passthrough", action="BUY",
                             created_at=now))
        s.add(ClaudeDecision(wallet_id=w.id, symbol="XRP-USD",
                             source="training_passthrough", action="SELL",
                             created_at=now))

    data = _call_endpoint()
    assert data["decisions"]["total"] == 5
    by_source = data["decisions"]["by_source"]
    assert "autonomous" in by_source
    assert by_source["autonomous"]["HOLD"] == 2
    assert by_source["autonomous"]["BUY"] == 1
    assert "training_passthrough" in by_source
    assert by_source["training_passthrough"]["BUY"] == 1
    assert by_source["training_passthrough"]["SELL"] == 1


def test_trades_grouped_by_calibration_source():
    """When PaperTrade rows have calibration metadata, the scorecard
    groups them by source so the operator can see how many trades were
    backed by measured data vs raw confidence."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    with session_scope() as s:
        w = Wallet(name="test", platform="paper",
                   bankroll_reset_at=now - timedelta(hours=2))
        s.add(w)
        s.flush()
        # 3 trades on knn_neighbors, 2 on exact_pattern, 1 on raw_confidence
        for n in (10, 15, 20):
            s.add(PaperTrade(wallet_id=w.id, symbol="BTC-USD", side="BUY",
                             qty=1.0, entry_price=50000.0, opened_at=now,
                             calibration_source="knn_neighbors",
                             calibration_sample_size=n))
        for n in (4, 6):
            s.add(PaperTrade(wallet_id=w.id, symbol="ETH-USD", side="BUY",
                             qty=1.0, entry_price=2000.0, opened_at=now,
                             calibration_source="exact_pattern",
                             calibration_sample_size=n))
        s.add(PaperTrade(wallet_id=w.id, symbol="SOL-USD", side="SELL",
                         qty=1.0, entry_price=86.0, opened_at=now,
                         calibration_source="raw_confidence",
                         calibration_sample_size=0))

    data = _call_endpoint()
    assert data["trades"]["total"] == 6
    breakdown = {r["source"]: r for r in data["trades"]["by_calibration"]}
    assert breakdown["knn_neighbors"]["count"] == 3
    assert breakdown["exact_pattern"]["count"] == 2
    assert breakdown["raw_confidence"]["count"] == 1
    # Average sample size sanity
    assert breakdown["knn_neighbors"]["avg_sample_size"] == 15.0  # (10+15+20)/3
    assert breakdown["exact_pattern"]["avg_sample_size"] == 5.0   # (4+6)/2


def test_calibration_breakdown_is_sorted_stable():
    """The UI renders the calibration breakdown in the same order every
    poll: exact_pattern → knn_neighbors → raw_confidence. This stable
    ordering keeps the bars from jumping around as new trades land."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    with session_scope() as s:
        w = Wallet(name="test", platform="paper",
                   bankroll_reset_at=now - timedelta(hours=2))
        s.add(w)
        s.flush()
        # Insert in REVERSE order — endpoint should still sort correctly.
        s.add(PaperTrade(wallet_id=w.id, symbol="A-USD", side="BUY",
                         qty=1.0, entry_price=1.0, opened_at=now,
                         calibration_source="raw_confidence", calibration_sample_size=0))
        s.add(PaperTrade(wallet_id=w.id, symbol="B-USD", side="BUY",
                         qty=1.0, entry_price=1.0, opened_at=now,
                         calibration_source="knn_neighbors", calibration_sample_size=10))
        s.add(PaperTrade(wallet_id=w.id, symbol="C-USD", side="BUY",
                         qty=1.0, entry_price=1.0, opened_at=now,
                         calibration_source="exact_pattern", calibration_sample_size=5))

    data = _call_endpoint()
    sources = [r["source"] for r in data["trades"]["by_calibration"]]
    assert sources == ["exact_pattern", "knn_neighbors", "raw_confidence"], (
        f"calibration breakdown order is unstable: got {sources}"
    )


def test_reflection_dedup_parsed_from_activity_log():
    """Reflection counts come from ActivityLog messages with the format
    'Reflection saved for trade #N: ... (new=N, reinforced=M)'.
    The endpoint parses the new/reinforced numbers and computes the
    dedup ratio. This is the metric that PROVES the new dedup is working
    — it should rise above 0 after the dedup commit."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    with session_scope() as s:
        w = Wallet(name="test", platform="paper",
                   bankroll_reset_at=now - timedelta(hours=2))
        s.add(w)
        s.flush()
        # Three reflections: 5 new / 0 reinforced, 3 new / 2 reinforced, 1 new / 4 reinforced.
        s.add(ActivityLog(category="ai", level="info", created_at=now,
            message="Reflection saved for trade #1: verdict=bad_call, score=-0.5, "
                    "lessons=5 (new=5, reinforced=0)"))
        s.add(ActivityLog(category="ai", level="info", created_at=now,
            message="Reflection saved for trade #2: verdict=bad_call, score=-0.5, "
                    "lessons=5 (new=3, reinforced=2)"))
        s.add(ActivityLog(category="ai", level="info", created_at=now,
            message="Reflection saved for trade #3: verdict=lucky, score=0.3, "
                    "lessons=5 (new=1, reinforced=4)"))

    data = _call_endpoint()
    refl = data["reflections"]
    assert refl["total"] == 3
    assert refl["lessons_new"] == 9        # 5 + 3 + 1
    assert refl["lessons_reinforced"] == 6  # 0 + 2 + 4
    # dedup_ratio = reinforced / (new + reinforced) = 6 / 15 = 0.40
    assert refl["dedup_ratio"] == pytest.approx(0.40, abs=0.01)


def test_session_cutoff_filters_old_rows():
    """Rows from before the cutoff (e.g. yesterday's trades, decisions from
    the prior session) must be excluded. This is what makes the scorecard
    a 'today' view, not all-time."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    yesterday = now - timedelta(days=1)
    with session_scope() as s:
        w = Wallet(name="test", platform="paper",
                   bankroll_reset_at=now - timedelta(minutes=30))
        s.add(w)
        s.flush()
        # One decision from yesterday (should be excluded).
        s.add(ClaudeDecision(wallet_id=w.id, symbol="OLD-USD",
                             source="autonomous", action="BUY",
                             created_at=yesterday))
        # One decision from now (should be included).
        s.add(ClaudeDecision(wallet_id=w.id, symbol="NEW-USD",
                             source="autonomous", action="HOLD",
                             created_at=now))

    data = _call_endpoint()
    assert data["decisions"]["total"] == 1, (
        f"old decision was not excluded by cutoff: {data}"
    )


def test_empty_reflections_are_counted_separately():
    """Reflections that fall into the 'lessons=0' failure path should be
    counted in the `empty` field so the operator can spot when Claude
    starts returning empty/unparseable reflections again."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    with session_scope() as s:
        w = Wallet(name="test", platform="paper",
                   bankroll_reset_at=now - timedelta(hours=2))
        s.add(w)
        s.flush()
        # One healthy reflection, two empty ones (Phase A observability fix).
        s.add(ActivityLog(category="ai", level="info", created_at=now,
            message="Reflection saved for trade #1: verdict=bad_call, score=-0.5, "
                    "lessons=5 (new=3, reinforced=2)"))
        s.add(ActivityLog(category="ai", level="warn", created_at=now,
            message="Reflection saved for trade #2: verdict=neutral, score=0.00, "
                    "lessons=0 (new=0, reinforced=0) — Could not parse Claude reflection."))
        s.add(ActivityLog(category="ai", level="warn", created_at=now,
            message="Reflection saved for trade #3: verdict=neutral, score=0.00, "
                    "lessons=0 (new=0, reinforced=0) — Claude unavailable: timeout"))

    data = _call_endpoint()
    assert data["reflections"]["total"] == 3
    assert data["reflections"]["empty"] == 2
