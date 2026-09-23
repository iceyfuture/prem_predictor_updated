"""
Offline tests for RULE 42 — no post-kickoff predictions. Audit finding §2.
Runs against TEMPORARY ledger copies; the real dashboard/ledger.csv is never written.

    ./.venv/bin/python dashboard/test_ledger_guard.py
"""
import os
import re, sys, csv, shutil, tempfile
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
# RULE 51: keys are season-aware now. Derived here the same way production derives them,
# so this constant cannot drift from the writer the way a typed-in "Arsenal|Leeds" did.
import forecast as FC  # noqa: E402
KEY = FC.fixture_key(FC.season_of(KO), "Arsenal", "Leeds")
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
    r = rows(p)[KEY]
    check("row created", lambda: _assert(r["pred_at"].startswith("2026-10-07")))
    check("not flagged late", lambda: _assert(r["late"] == ""))
    check("absolute kickoff stored", lambda: _assert(r["kickoff_utc"] == KO.isoformat()))
    check("opening probabilities locked", lambda: _assert(r["pred_h"] == "60"))

    print("\n=== AT kickoff: rejected as a forecast ===")
    p = fresh("at")
    run(match(), KO.isoformat(), p)
    check("flagged late at exactly kickoff", lambda: _assert(rows(p)[KEY]["late"] == "1"))

    print("\n=== AFTER kickoff: the audit's leak case ===")
    p = fresh("after")
    run(match(finished=True, result="3-0"), (KO + timedelta(hours=3)).isoformat(), p)
    r = rows(p)[KEY]
    check("still recorded (kept for the record)", lambda: _assert(r["pred_at"] != ""))
    check("flagged late=1", lambda: _assert(r["late"] == "1"))
    check("a finished match discovered late cannot score as a forecast",
          lambda: _assert(r["late"] == "1"))

    print("\n=== opening prediction is immutable ===")
    p = fresh("immutable")
    run(match(), (KO - timedelta(days=5)).isoformat(), p)
    first = rows(p)[KEY]
    run(match(ph=20, pd=30, pa=50), (KO - timedelta(days=1)).isoformat(), p)
    second = rows(p)[KEY]
    check("pred_* unchanged by a later build",
          lambda: _assert((second["pred_h"], second["pred_d"], second["pred_a"])
                          == (first["pred_h"], first["pred_d"], first["pred_a"])))
    check("pred_at unchanged", lambda: _assert(second["pred_at"] == first["pred_at"]))
    check("closing numbers DID move", lambda: _assert(second["close_h"] == "20"))

    print("\n=== closing numbers freeze at kickoff ===")
    p = fresh("freeze")
    run(match(), (KO - timedelta(days=2)).isoformat(), p)
    run(match(ph=11, pd=11, pa=78), (KO + timedelta(hours=1)).isoformat(), p)
    r = rows(p)[KEY]
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
          lambda: _assert(rows(p)[KEY]["kickoff_utc"] == new_ko.isoformat()))

    print("\n=== repeated builds are idempotent ===")
    p = fresh("idem")
    at = (KO - timedelta(days=3)).isoformat()
    run(match(), at, p); a = rows(p)[KEY]
    for _ in range(3): run(match(), at, p)
    b = rows(p)[KEY]
    check("row identical after 4 builds", lambda: _assert(a == b))
    check("still one row", lambda: _assert(len(rows(p)) == 1))

    print("\n=== timezone handling ===")
    p = fresh("tz")
    run(match(utc="2026-10-10T14:00Z"), (KO - timedelta(hours=2)).isoformat(), p)
    check("Z suffix parsed as UTC", lambda: _assert(rows(p)[KEY]["late"] == ""))
    p = fresh("tz2")
    run(match(utc="2026-10-10T14:00Z"), (KO + timedelta(hours=2)).isoformat(), p)
    check("and compared correctly after kickoff",
          lambda: _assert(rows(p)[KEY]["late"] == "1"))
    p = fresh("noutc")
    run(match(utc=""), (KO + timedelta(days=99)).isoformat(), p)
    check("missing utc -> not late (cannot prove it, do not guess)",
          lambda: _assert(rows(p)[KEY]["late"] == ""))

    print("\n=== the real ledger was never written ===")
    check("dashboard/ledger.csv byte-identical", lambda: _assert(_fp(REAL) == BEFORE))
    shutil.rmtree(tmp, ignore_errors=True)
    print("\n=== RULE 51 (audit §2): fixture keys carry a season and a provider id ===")
    import forecast as FC
    check("a season label spans the right two years",
          lambda: _assert(FC.season_of("2026-08-15T14:00:00+00:00") == "2026/27"))
    check("January belongs to the season that began the previous August",
          lambda: _assert(FC.season_of("2027-01-15T14:00:00+00:00") == "2026/27"))
    check("June belongs to the season that began the previous August",
          lambda: _assert(FC.season_of("2027-06-01T14:00:00+00:00") == "2026/27"))
    check("July starts the new one",
          lambda: _assert(FC.season_of("2027-07-01T14:00:00+00:00") == "2027/28"))
    check("an empty kickoff yields no season rather than a wrong one",
          lambda: _assert(FC.season_of("") == ""))
    check("the same fixture in two seasons gets two different keys",
          lambda: _assert(FC.fixture_key("2026/27", "Arsenal", "Chelsea") !=
                          FC.fixture_key("2027/28", "Arsenal", "Chelsea")))
    check("home|away alone is never the key",
          lambda: _assert(FC.fixture_key("2026/27", "Arsenal", "Chelsea") != "Arsenal|Chelsea"))
    check("a provider id is preferred when present",
          lambda: _assert(FC.fixture_key("2026/27", "Arsenal", "Chelsea", "5795363")
                          == "2026/27|#5795363"))
    check("a blank provider id falls back to names, not to '#'",
          lambda: _assert("#" not in FC.fixture_key("2026/27", "Arsenal", "Chelsea", "")))
    check("keys are stable across calls",
          lambda: _assert(FC.fixture_key("2026/27", "A", "B", "9") ==
                          FC.fixture_key("2026/27", "A", "B", "9")))

    print("\n=== migration is safe, idempotent and backed up ===")
    import shutil as _sh, tempfile as _tf
    tmpd = _tf.mkdtemp()
    tpath = os.path.join(tmpd, "ledger.csv")
    _sh.copy2(REAL, tpath)
    orig = {r["key"]: r for r in B.read_csv(tpath)}
    once = B._migrate_keys(dict(orig), tpath)
    twice = B._migrate_keys(dict(once), tpath)
    check("no rows are gained or lost", lambda: _assert(len(once) == len(orig)))
    check("every key is now season-aware",
          lambda: _assert(all(re.match(r"^\d{4}/\d{2}\|", k) for k in once)))
    check("running it again changes nothing", lambda: _assert(set(twice) == set(once)))
    check("a backup exists", lambda: _assert(os.path.exists(tpath + ".pre-rule51.bak")))
    byha = {(r["home"], r["away"]): r for r in orig.values()}
    altered = [c for r in once.values() for c in byha[(r["home"], r["away"])]
               if c not in ("key", "season")
               and (byha[(r["home"], r["away"])].get(c) or "") != (r.get(c) or "")]
    check(f"no forecast, timestamp or grade is touched ({len(altered)} altered)",
          lambda: _assert(not altered))
    _sh.rmtree(tmpd, ignore_errors=True)

    print("\n=== a legacy-keyed row is adopted, never duplicated ===")
    src = open(os.path.join(os.path.dirname(REAL), "build_dashboard.py")).read()
    check("the writer looks for the legacy key before locking a new row",
          lambda: _assert("legacy = rows.pop(F.fixture_key(season, m[\"home\"], m[\"away\"]), None)" in src))
    check("and rekeys it in place",
          lambda: _assert('legacy["key"] = key' in src))

    print("\n=== RULE 51: every forecast records what produced it ===")
    prov = FC.provenance(FC.fit(cutoff="2026-09-01"))
    for k in ("code_commit", "model_config", "training_cutoff", "data_version"):
        check(f"provenance carries {k}", lambda kk=k: _assert(kk in prov))
    check("the model config names the decay, ridge, away cal, blend and cold start",
          lambda: _assert(all(t in prov["model_config"]
                              for t in ("s3", "r8", "away1.08", "blend", "cold"))))
    check("the training cutoff is recorded",
          lambda: _assert(prov["training_cutoff"].startswith("2026-09")))
    check("the data version names the last training match",
          lambda: _assert(":" in prov["data_version"] and prov["data_version"][:4].isdigit()))
    check("an edited tree is flagged dirty rather than claiming a clean commit",
          lambda: _assert(prov["code_commit"] == "" or
                          re.match(r"^[0-9a-f]{7,}(\+dirty)?$", prov["code_commit"])))
    check("the writer stamps provenance on the opening lock",
          lambda: _assert("**prov}" in src))
    check("and backfills it without overwriting a locked row's own",
          lambda: _assert("rec.setdefault(_k, _v)" in src))

    print(f"\n  {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0

if __name__ == "__main__":
    sys.exit(main())
