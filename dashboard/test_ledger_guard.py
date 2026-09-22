"""
Offline tests for RULE 42 — no post-kickoff predictions. Audit finding §2.
Runs against TEMPORARY ledger copies; the real dashboard/ledger.csv is never written.

    ./.venv/bin/python dashboard/test_ledger_guard.py
"""
import os, sys, csv, shutil, tempfile
from datetime import datetime, timedelta, timezone
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_dashboard as B

PASS = FAIL = 0
REAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ledger.csv")
def _fp(p):
    if not os.path.exists(p): return None
    st = os.stat(p)
    with open(p) as f: return (st.st_size, st.st_mtime, f.read())
BEFORE = _fp(REAL)

def check(name, fn, expect=None):
    global PASS, FAIL
    try: fn()
    except Exception as e:
        if expect and isinstance(e, expect): print(f"  ok   {name}"); PASS += 1
        else: print(f"  FAIL {name}: {type(e).__name__}: {e}"); FAIL += 1
        return
    if expect: print(f"  FAIL {name}: expected {expect.__name__}"); FAIL += 1
    else: print(f"  ok   {name}"); PASS += 1
def _assert(c):
    if not c: raise AssertionError("assertion failed")

KO = datetime(2026, 10, 10, 14, 0, tzinfo=timezone.utc)
def match(**kw):
    m = {"home": "Arsenal", "away": "Leeds", "time": "Sat 10 Oct 14:00",
         "utc": KO.isoformat(), "ph": 60, "pd": 23, "pa": 17,
         "finished": False, "live": False, "result": None, "kalshi": None,
         "mkt": None}
    m.update(kw); return m
def weeks(m): return [{"gw": 6, "matches": [m]}]

def run(m, at, path):
    """record_ledger writes to HERE/ledger.csv - point HERE at a temp dir."""
    saved = B.HERE
    B.HERE = path
    try: return B.record_ledger(weeks(m), at)
    finally: B.HERE = saved

def rows(path):
    p = os.path.join(path, "ledger.csv")
    if not os.path.exists(p): return {}
    with open(p) as f: return {r["key"]: r for r in csv.DictReader(f)}

def main():
    tmp = tempfile.mkdtemp(prefix="ledger_guard_")
    d = lambda n: os.path.join(tmp, n) if not os.makedirs(os.path.join(tmp, n), exist_ok=True) else None
    def fresh(n):
        p = os.path.join(tmp, n); os.makedirs(p, exist_ok=True); return p

    print("=== before kickoff: a real forecast ===")
    p = fresh("before")
    run(match(), (KO - timedelta(days=3)).isoformat(), p)
    r = rows(p)["Arsenal|Leeds"]
    check("row created", lambda: _assert(r["pred_at"].startswith("2026-10-07")))
    check("not flagged late", lambda: _assert(r["late"] == ""))
    check("absolute kickoff stored", lambda: _assert(r["kickoff_utc"] == KO.isoformat()))
    check("opening probabilities locked", lambda: _assert(r["pred_h"] == "60"))

    print("\n=== AT kickoff: rejected as a forecast ===")
    p = fresh("at")
    run(match(), KO.isoformat(), p)
    check("flagged late at exactly kickoff", lambda: _assert(rows(p)["Arsenal|Leeds"]["late"] == "1"))

    print("\n=== AFTER kickoff: the audit's leak case ===")
    p = fresh("after")
    run(match(finished=True, result="3-0"), (KO + timedelta(hours=3)).isoformat(), p)
    r = rows(p)["Arsenal|Leeds"]
    check("still recorded (kept for the record)", lambda: _assert(r["pred_at"] != ""))
    check("flagged late=1", lambda: _assert(r["late"] == "1"))
    check("a finished match discovered late cannot score as a forecast",
          lambda: _assert(r["late"] == "1"))

    print("\n=== opening prediction is immutable ===")
    p = fresh("immutable")
    run(match(), (KO - timedelta(days=5)).isoformat(), p)
    first = rows(p)["Arsenal|Leeds"]
    run(match(ph=20, pd=30, pa=50), (KO - timedelta(days=1)).isoformat(), p)
    second = rows(p)["Arsenal|Leeds"]
    check("pred_* unchanged by a later build",
          lambda: _assert((second["pred_h"], second["pred_d"], second["pred_a"])
                          == (first["pred_h"], first["pred_d"], first["pred_a"])))
    check("pred_at unchanged", lambda: _assert(second["pred_at"] == first["pred_at"]))
    check("closing numbers DID move", lambda: _assert(second["close_h"] == "20"))

    print("\n=== closing numbers freeze at kickoff ===")
    p = fresh("freeze")
    run(match(), (KO - timedelta(days=2)).isoformat(), p)
    run(match(ph=11, pd=11, pa=78), (KO + timedelta(hours=1)).isoformat(), p)
    r = rows(p)["Arsenal|Leeds"]
    check("closing number did not move after kickoff",
          lambda: _assert(r["close_h"] != "11"))
    check("even though the feed still said not-live",
          lambda: _assert(r["late"] == ""))

    print("\n=== rescheduled fixture keeps one row ===")
    p = fresh("resched")
    run(match(), (KO - timedelta(days=4)).isoformat(), p)
    new_ko = KO + timedelta(days=1)
    run(match(utc=new_ko.isoformat(), time="Sun 11 Oct 14:00"),
        (KO - timedelta(days=2)).isoformat(), p)
    check("still one row", lambda: _assert(len(rows(p)) == 1))
    check("kickoff_utc follows the reschedule",
          lambda: _assert(rows(p)["Arsenal|Leeds"]["kickoff_utc"] == new_ko.isoformat()))

    print("\n=== repeated builds are idempotent ===")
    p = fresh("idem")
    at = (KO - timedelta(days=3)).isoformat()
    run(match(), at, p); a = rows(p)["Arsenal|Leeds"]
    for _ in range(3): run(match(), at, p)
    b = rows(p)["Arsenal|Leeds"]
    check("row identical after 4 builds", lambda: _assert(a == b))
    check("still one row", lambda: _assert(len(rows(p)) == 1))

    print("\n=== timezone handling ===")
    p = fresh("tz")
    run(match(utc="2026-10-10T14:00Z"), (KO - timedelta(hours=2)).isoformat(), p)
    check("Z suffix parsed as UTC", lambda: _assert(rows(p)["Arsenal|Leeds"]["late"] == ""))
    p = fresh("tz2")
    run(match(utc="2026-10-10T14:00Z"), (KO + timedelta(hours=2)).isoformat(), p)
    check("and compared correctly after kickoff",
          lambda: _assert(rows(p)["Arsenal|Leeds"]["late"] == "1"))
    p = fresh("noutc")
    run(match(utc=""), (KO + timedelta(days=99)).isoformat(), p)
    check("missing utc -> not late (cannot prove it, do not guess)",
          lambda: _assert(rows(p)["Arsenal|Leeds"]["late"] == ""))

    print("\n=== the real ledger was never written ===")
    check("dashboard/ledger.csv byte-identical", lambda: _assert(_fp(REAL) == BEFORE))
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n  {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0

if __name__ == "__main__":
    sys.exit(main())
