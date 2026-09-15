#!/usr/bin/env python3
"""
run_council.py — the standalone AI Council runner.

Finds upcoming Premier League fixtures from the SAME data the dashboard uses, asks the ledger
which of them already have a locked forecast, and runs the Council on the rest.

DRY RUN IS THE DEFAULT. Without --live this script cannot call Claude and cannot touch the
ledger: it prints what it WOULD do and exits. That default is deliberate. A full Council run
is four subprocess calls to Claude per fixture (~70s measured), so a script that predicts by
accident is both slow and expensive, and an accidental row in an append-only forward-test
ledger cannot be taken back - the first prediction is the one that counts, forever.

    python dashboard/run_council.py                     # dry run over the next 7 days
    python dashboard/run_council.py --days 3            # shorter horizon
    python dashboard/run_council.py --fixture "A|B"     # one fixture
    python dashboard/run_council.py --live --limit 1    # actually run, at most one fixture

WHY IT IS STANDALONE
The Council is an experimental layer with no demonstrated edge, and at ~70s per fixture it
would add roughly twelve minutes to a dashboard build that currently runs in seconds. Wiring
it into build_dashboard.py would make the desk's refresh depend on a subprocess pipeline that
can fail in ways the desk cannot recover from. Running it separately keeps the blast radius
at "no Council rows this week".
"""
import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import council as C                      # noqa: E402  (MatchContext, predict)
import council_ledger as L               # noqa: E402
import council_reasoning as CR           # noqa: E402

PAYLOAD = os.path.join(HERE, "dashboard.json")


def load_payload(path=None):
    """The desk's own built payload - model probabilities, market, xG, form, team news."""
    import json
    with open(path or PAYLOAD) as f:
        return json.load(f)


def load_events():
    """The fixture feed, for the one thing the payload drops: a parseable UTC kickoff.

    dashboard.json stores kickoff as "Fri 18 Sep 19:00" - no year, no zone - which cannot be
    compared to a horizon without guessing. feeds.espn_events() carries `utc`, and it is the
    same loader build_dashboard uses, so the two cannot disagree about which fixtures exist.
    """
    import feeds
    return feeds.espn_events()


def _utc(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def build_context(match, kickoff=None):
    """The same MatchContext a real Council run uses. One place, so the runner and any future
    caller cannot drift into showing the seats different evidence."""
    mk = (match.get("mkt") or {}).get("imp") or {}
    kal = (match.get("kalshi") or {}).get("mid") or {}
    xg = match.get("xg")
    hxg, axg = ([float(x) for x in xg.split(" - ")] if xg else [None, None])
    form = match.get("form") if isinstance(match.get("form"), dict) else {}
    news = match.get("news") if isinstance(match.get("news"), dict) else {}
    items = lambda side: [f"{i.get('who')}: {i.get('what')}"
                          for i in (news.get(side) or []) if i.get("what")]
    pct = lambda v: (v / 100.0) if v not in (None, "") else None
    return C.MatchContext(
        home_team=match["home"], away_team=match["away"],
        kickoff=kickoff or match.get("time"),
        model_home=pct(match.get("ph")), model_draw=pct(match.get("pd")),
        model_away=pct(match.get("pa")),
        expected_home_goals=hxg, expected_away_goals=axg,
        market_home=mk.get("h"), market_draw=mk.get("d"), market_away=mk.get("a"),
        kalshi_home=pct(kal.get("h")), kalshi_draw=pct(kal.get("d")),
        kalshi_away=pct(kal.get("a")),
        home_form=form.get("h"), away_form=form.get("a"),
        home_news=items("h"), away_news=items("a"))


def upcoming(days=7, now=None, payload=None, events=None, fixture=None):
    """Fixtures that have NOT kicked off and start within `days`. Newest kickoff last.

    A fixture already under way is excluded rather than locked late: a late row is kept for
    the record but scores nothing, so spending four Claude calls to create one is pure waste.
    """
    now = now or datetime.now(timezone.utc)
    payload = payload if payload is not None else load_payload()
    events = events if events is not None else load_events()
    horizon = now + timedelta(days=days)

    ko = {}
    for e in events:
        t = _utc(e.get("utc"))
        if t:
            ko[(e.get("home"), e.get("away"))] = t

    out = []
    for wk in payload.get("weeks", []):
        for m in wk.get("matches", []):
            if m.get("finished") or m.get("live"):
                continue
            key = f"{m['home']}|{m['away']}"
            if fixture and key != fixture:
                continue
            t = ko.get((m["home"], m["away"]))
            if t is None or t <= now or t > horizon:
                continue
            out.append({"key": key, "match": m, "kickoff": t,
                        "gw": wk.get("gw"), "when": m.get("time")})
    out.sort(key=lambda r: (r["kickoff"], r["key"]))
    return out


def run(days=7, live=False, fixture=None, limit=None, now=None,
        payload=None, events=None, ledger_path=None, reasoning_path=None, out=print):
    """Decide, and (only with live=True) act. Returns a summary dict."""
    rows = upcoming(days=days, now=now, payload=payload, events=events, fixture=fixture)
    out("AI Council runner" + ("" if live else "   [DRY RUN - no Claude calls, no writes]"))
    out("")

    locked_n = would_run = ran = failed = reasoning_failed = 0
    errors = []
    for r in rows:
        out(f"{r['match']['home']} vs {r['match']['away']}"
            f"   (GW{r['gw']}, {r['when']})")
        if L.is_locked(r["key"], ledger_path):
            locked_n += 1
            out("  LOCKED -> skip")
            continue
        if not live:
            would_run += 1
            out("  NOT LOCKED -> would run Council")
            continue
        if limit is not None and ran >= limit:
            would_run += 1
            out(f"  NOT LOCKED -> skipped, --limit {limit} reached")
            continue

        # RE-CHECK immediately before spending four Claude calls. `upcoming` was computed
        # earlier and another runner (or a previous fixture in this same loop, for a repeated
        # fixture) may have locked it since. The ledger is the only authority.
        if L.is_locked(r["key"], ledger_path):
            locked_n += 1
            out("  LOCKED (raced) -> skip")
            continue
        ctx = build_context(r["match"], kickoff=r["kickoff"].isoformat(timespec="minutes"))
        try:
            prediction = C.predict(ctx)
        except Exception as e:                      # keep going; one bad fixture is not fatal
            failed += 1
            errors.append((r["key"], f"{type(e).__name__}: {e}"))
            out(f"  FAILED -> {type(e).__name__}: {e}")
            out("  nothing written for this fixture")
            continue
        try:
            row = L.record(ctx, prediction, path=ledger_path)
        except Exception as e:
            # The lock failed, so this forecast does not exist as far as the desk is
            # concerned. Writing reasoning for it would leave an orphan explaining a
            # prediction no ledger row claims.
            failed += 1
            errors.append((r["key"], f"ledger lock failed: {type(e).__name__}: {e}"))
            out(f"  FAILED -> could not lock: {type(e).__name__}: {e}")
            out("  no reasoning written")
            continue
        ran += 1
        h, d, a = prediction.probabilities()
        out(f"  RAN -> council {h:.2f}/{d:.2f}/{a:.2f} {prediction.predicted_outcome}"
            f"  consensus {prediction.consensus_score}"
            f"{'  [LATE]' if row.get('late') else ''}")

        # The ledger row is locked and correct from here on. A reasoning failure is a
        # SIDECAR problem: report it loudly, but never re-run Claude (four more calls for a
        # forecast we already have) and never touch the locked row to "fix" it.
        try:
            wrote = CR.append(ctx, prediction, locked_at=row.get("locked_at"),
                              path=reasoning_path)
            if wrote is None:
                out("  reasoning already recorded for this fixture - not duplicated")
        except Exception as e:
            reasoning_failed += 1
            errors.append((r["key"], f"reasoning not saved: {type(e).__name__}: {e}"))
            out(f"  WARNING -> forecast is LOCKED but reasoning was not saved: "
                f"{type(e).__name__}: {e}")
            out("  the ledger row stands; Claude was NOT re-run")

    out("")
    out("Summary:")
    out(f"  upcoming: {len(rows)}")
    out(f"  locked: {locked_n}")
    out(f"  {'ran' if live else 'would_run'}: {ran if live else would_run}")
    if live and reasoning_failed:
        out(f"  reasoning not saved: {reasoning_failed} (forecasts still locked)")
    if live and failed:
        out(f"  failed: {failed}")
        for k, msg in errors:
            out(f"    {k}: {msg}")
    return {"upcoming": len(rows), "locked": locked_n, "would_run": would_run,
            "ran": ran, "failed": failed, "reasoning_failed": reasoning_failed,
            "errors": errors,
            "keys": [r["key"] for r in rows], "live": live}


def main(argv=None):
    p = argparse.ArgumentParser(description="Run the AI Council over upcoming fixtures.")
    p.add_argument("--days", type=int, default=7,
                   help="only fixtures kicking off within this many days (default 7)")
    p.add_argument("--live", action="store_true",
                   help="actually call Claude and write to the ledger (default: dry run)")
    p.add_argument("--fixture", metavar="HOME|AWAY",
                   help='run one fixture only, e.g. --fixture "Brentford|Chelsea"')
    p.add_argument("--limit", type=int,
                   help="run at most N NEW fixtures (locked ones never count)")
    args = p.parse_args(argv)
    s = run(days=args.days, live=args.live, fixture=args.fixture, limit=args.limit)
    return 1 if s["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
