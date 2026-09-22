"""
Offline tests for the ESPN odds loader and the Council market-availability guard.
No network: every ESPN response is a canned fixture. No Claude calls.

    ./.venv/bin/python dashboard/test_espn_odds.py
"""
import os, sys, json, urllib.error
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feeds, council as C

PASS = FAIL = 0


def check(name, fn, expect=None):
    global PASS, FAIL
    try:
        fn()
    except Exception as e:
        if expect and isinstance(e, expect):
            print(f"  ok   {name} -> {type(e).__name__}"); PASS += 1
        else:
            print(f"  FAIL {name}: {type(e).__name__}: {e}"); FAIL += 1
        return
    if expect:
        print(f"  FAIL {name}: expected {expect.__name__}"); FAIL += 1
    else:
        print(f"  ok   {name}"); PASS += 1


def _assert(c):
    if not c:
        raise AssertionError("assertion failed")


def comp(home="Arsenal", away="Leeds United", odds=True, ml=(-280, 390, 700)):
    c = {"id": "1", "competitors": [
            {"homeAway": "home", "team": {"displayName": home, "abbreviation": "ARS",
                                          "color": "EF0107"}, "score": "0"},
            {"homeAway": "away", "team": {"displayName": away, "abbreviation": "LEE",
                                          "color": "FFFFFF"}, "score": "0"}],
         "venue": {"fullName": "Emirates"}}
    if odds:
        # the shape parse_odds actually reads: odds[0].moneyline.<side>.close.odds
        leg = lambda v: {"close": {"odds": str(v)}}
        c["odds"] = [{"provider": {"displayName": "DraftKings"}, "overUnder": 2.5,
                      "moneyline": {"home": leg(ml[0]), "draw": leg(ml[1]),
                                    "away": leg(ml[2])}}]
    return c


def day(events):
    return {"events": [{"id": str(100+i), "date": "2026-10-10T11:30Z",
                        "status": {"type": {"name": "STATUS_SCHEDULED"}},
                        "competitions": [c]} for i, c in enumerate(events)]}


def with_get(responses, fn):
    """Replace feeds._get with a canned map of day -> response | Exception."""
    saved = feeds._get
    def fake(url, cache_key, max_age_min=180):
        d = url.split("dates=")[1].split("&")[0]
        r = responses.get(d)
        if isinstance(r, Exception):
            raise r
        if r is None:
            raise urllib.error.HTTPError(url, 400, "Bad Request", {}, None)
        return r
    feeds._get = fake
    try:
        return fn()
    finally:
        feeds._get = saved


def main():
    print("=== the fix: single days, never ranges ===")
    days = feeds._espn_days("20261010", "20261012")
    check("expands a window into single days",
          lambda: _assert(days == ["20261010", "20261011", "20261012"]))
    check("no value contains a range separator",
          lambda: _assert(not any("-" in d for d in days)))
    check("explicit days are used verbatim",
          lambda: _assert(feeds._espn_days(days=["20261010"]) == ["20261010"]))
    check("explicit days are de-duplicated and sorted",
          lambda: _assert(feeds._espn_days(days=["20261012","20261010","20261010"])
                          == ["20261010","20261012"]))
    check("the URL template takes one date",
          lambda: _assert("dates=20261010&" in feeds.ESPN.format("20261010")))

    print("\n=== a valid odds response ===")
    got = with_get({"20261010": day([comp()])},
                   lambda: feeds._espn_events_raw(days=["20261010"]))
    check("one fixture parsed", lambda: _assert(len(got) == 1))
    check("teams mapped", lambda: _assert((got[0]["home"], got[0]["away"]) == ("Arsenal","Leeds")))
    o = got[0]["odds"]
    check("odds present", lambda: _assert(o is not None))
    check("provider carried", lambda: _assert(o["provider"] == "DraftKings"))
    check("decimal odds computed from american moneyline",
          lambda: _assert(o["dec"]["h"] == round(100/280 + 1, 3)))
    check("implied probabilities sum to ~1 (vig removed)",
          lambda: _assert(abs(sum(o["imp"].values()) - 1.0) < 0.005))
    check("overround reported and positive", lambda: _assert(o["overround"] > 0))
    check("favourite has the highest implied probability",
          lambda: _assert(o["imp"]["h"] > o["imp"]["d"] and o["imp"]["h"] > o["imp"]["a"]))
    check("over/under line carried", lambda: _assert(o["ou_line"] == 2.5))

    print("\n=== failures degrade, they do not crash ===")
    got = with_get({"20261010": None},   # HTTP 400
                   lambda: feeds._espn_events_raw(days=["20261010"]))
    check("HTTP 400 -> no fixtures, no exception", lambda: _assert(got == []))
    got = with_get({"20261010": day([comp(odds=False)])},
                   lambda: feeds._espn_events_raw(days=["20261010"]))
    check("no odds offered -> fixture still returned", lambda: _assert(len(got) == 1))
    check("no odds offered -> odds is None", lambda: _assert(got[0]["odds"] is None))
    got = with_get({"20261010": {"events": [{"id": "1", "competitions": [{}]}]}},
                   lambda: feeds._espn_events_raw(days=["20261010"]))
    check("malformed event skipped, no crash", lambda: _assert(got == []))
    got = with_get({"20261010": {"garbage": True}},
                   lambda: feeds._espn_events_raw(days=["20261010"]))
    check("response with no events key -> empty", lambda: _assert(got == []))

    print("\n=== one bad day does not break the others ===")
    got = with_get({"20261010": None,                       # 400
                    "20261011": day([comp("Chelsea", "AFC Bournemouth")]),
                    "20261012": RuntimeError("network down")},
                   lambda: feeds._espn_events_raw(days=["20261010","20261011","20261012"]))
    check("the healthy day still returns its fixture", lambda: _assert(len(got) == 1))
    check("and it is the right one",
          lambda: _assert((got[0]["home"], got[0]["away"]) == ("Chelsea","Bournemouth")))

    print("\n=== the overlay keeps fixtures when odds fail ===")
    fx = [{"home":"Arsenal","away":"Leeds","utc":"2099-01-01T12:00Z","odds":None,
           "finished":False,"live":False}]
    out = with_get({}, lambda: feeds._overlay_odds(list(fx)))
    check("fixtures survive an unusable odds source", lambda: _assert(len(out) == 1))
    check("and stay unpriced rather than faked", lambda: _assert(out[0]["odds"] is None))
    past = [{"home":"A","away":"B","utc":"2020-01-01T12:00Z","odds":None,
             "finished":True,"live":False}]
    out = with_get({}, lambda: feeds._overlay_odds(list(past)))
    check("finished fixtures are not re-priced", lambda: _assert(out[0]["odds"] is None))

    print("\n=== has_usable_market() ===")
    mk = lambda **k: C.MatchContext("A","B", model_home=.4, model_draw=.3, model_away=.3, **k)
    both = mk(market_home=.5, market_draw=.3, market_away=.2,
              kalshi_home=.5, kalshi_draw=.3, kalshi_away=.2)
    book = mk(market_home=.5, market_draw=.3, market_away=.2)
    exch = mk(kalshi_home=.5, kalshi_draw=.3, kalshi_away=.2)
    none_ = mk()
    check("both sources -> usable", lambda: _assert(C.has_usable_market(both)))
    check("bookmaker only -> usable", lambda: _assert(C.has_usable_market(book)))
    check("exchange only -> usable", lambda: _assert(C.has_usable_market(exch)))
    check("neither -> NOT usable", lambda: _assert(not C.has_usable_market(none_)))
    check("partial bookmaker (one leg missing) -> not usable",
          lambda: _assert(not C.has_usable_market(mk(market_home=.5))))
    check("sources listed", lambda: _assert(C.market_sources(both) == ["bookmaker","exchange"]))
    check("exchange-only listed", lambda: _assert(C.market_sources(exch) == ["exchange"]))
    check("none listed", lambda: _assert(C.market_sources(none_) == []))
    check("rejects a non-MatchContext",
          lambda: C.has_usable_market({"a":1}), C.CouncilValidationError)
    check("the real Tottenham case (no book, no kalshi) would be skipped",
          lambda: _assert(not C.has_usable_market(
              C.MatchContext("Tottenham","Aston Villa",
                             model_home=.37, model_draw=.27, model_away=.36))))

    print(f"\n  {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
