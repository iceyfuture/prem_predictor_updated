"""
Offline guard for dashboard/index.html: the page's JavaScript must actually parse.

Why this exists: commit 626cb71 shipped `function renderNews();renderSourceHealth(){`,
a mangled edit that merged two declarations. A single SyntaxError anywhere in the one
inline <script> kills the WHOLE script, so every panel on the page silently rendered
nothing -- the build succeeded, the JSON was correct, and the site was dead. Nothing in
the suite caught it because no test ever parsed the page.

Uses JavaScriptCore (ships with macOS) when available; skips with a clear message
otherwise, so an absent engine is never mistaken for a passing test.

    ./.venv/bin/python dashboard/test_dashboard_html.py
"""
import os, re, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.join(HERE, "index.html")
JSC = ("/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc")

PASS = FAIL = SKIP = 0


def check(name, fn, expect=None):
    global PASS, FAIL
    try:
        fn()
    except Exception as e:
        if expect and isinstance(e, expect):
            print(f"  ok   {name}"); PASS += 1
        else:
            print(f"  FAIL {name}: {type(e).__name__}: {e}"); FAIL += 1
        return
    if expect:
        print(f"  FAIL {name}: expected {expect.__name__}"); FAIL += 1
    else:
        print(f"  ok   {name}"); PASS += 1


def _assert(c, msg="assertion failed"):
    if not c:
        raise AssertionError(msg)


def scripts(html):
    return re.findall(r"<script[^>]*>(.*?)</script>", html, re.S)


def parse_error(js):
    """Return None if `js` parses, else the engine's message."""
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(js); src = f.name
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write("try{new Function(readFile(%r));print('OK')}"
                "catch(e){print('ERR '+e.message)}" % src)
        chk = f.name
    try:
        out = subprocess.run([JSC, chk], capture_output=True, text=True, timeout=60).stdout.strip()
    finally:
        os.unlink(src); os.unlink(chk)
    return None if out == "OK" else (out[4:] if out.startswith("ERR ") else out or "no output")


def main():
    global SKIP
    html = open(INDEX, encoding="utf-8").read()
    blocks = scripts(html)

    print("=== the page has script to check ===")
    check("index.html contains at least one <script>", lambda: _assert(blocks))
    check("the main block is substantial (not a stub)",
          lambda: _assert(max((len(b) for b in blocks), default=0) > 10000))

    print("\n=== no mangled function declarations ===")
    bad = re.findall(r"function\s*[A-Za-z_$][\w$]*\s*\([^)]*\)\s*;", html)
    check(f"no `function name();` anywhere ({len(bad)} found)",
          lambda: _assert(not bad, f"mangled declarations: {bad[:3]}"))

    print("\n=== every inline script parses ===")
    if not os.path.exists(JSC):
        print(f"  SKIP no JavaScript engine at {JSC}")
        print("       (dependency missing, NOT a code failure -- install Xcode CLT or node)")
        SKIP += 1
    else:
        for i, js in enumerate(blocks):
            if len(js.strip()) < 40:
                continue
            err = parse_error(js)
            check(f"script block {i} parses" + (f" -- {err}" if err else ""),
                  lambda e=err: _assert(e is None, e or ""))

    print("\n=== functions the market panel depends on are declared ===")
    for fn in ("renderNews", "renderSourceHealth", "renderBacktest",
               "drawMarket", "marketPaired", "marketCaveat"):
        check(f"{fn} is declared once",
              lambda f=fn: _assert(len(re.findall(rf"function\s+{f}\s*\(", html)) == 1))

    print("\n=== the market caveat is not hard-coded ===")
    check("no frozen RPS figures in the Kalshi caveat",
          lambda: _assert("0.2061 vs 0.1953" not in html))
    check("the caveat reads DATA.backtest instead",
          lambda: _assert("DATA.backtest&&DATA.backtest.market" in html))

    print(f"\n  {PASS} passed, {FAIL} failed, {SKIP} skipped")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
