"""
Premier League match prediction CARD — the same style as the World Cup predictor:
favourite, win-probability bar, most-likely scorelines, and anytime goalscorers per team.

Under the hood it uses the strengthened model: Dixon-Coles blended with the goal-supremacy
recent-form odds model (0.75/0.25), plus fair odds (1/prob) and the squad-based scorer model.

  ~/prem_predictor/.venv/bin/python predict.py "Nott'm Forest" "Leeds"

First team is HOME, second AWAY. Team names use football-data.co.uk spellings
(see outputs/team_rankings.csv).
"""
import json
import os
import sys
import numpy as np

import prem_dixon_coles as dc
import supremacy_odds as so
import prem_scorer as ps

HERE = os.path.dirname(os.path.abspath(__file__))
BLEND = os.path.join(HERE, ".state", "blend.json")


def top_scores(model, home, away, k=3):
    M, _, _ = dc.score_matrix(model, home, away)
    flat = np.dstack(np.unravel_index(np.argsort(-M, axis=None), M.shape))[0]
    return [(int(i), int(j), float(M[i, j])) for i, j in flat[:k]]


def prob_bar(ph, pdw, pa, n=28):
    a = round(ph * n); d = round(pdw * n)
    return "█" * a + "░" * d + "▒" * max(n - a - d, 0)


def main():
    if len(sys.argv) != 3:
        print(__doc__); return
    home, away = sys.argv[1], sys.argv[2]
    dc_model = dc.get_model()
    mapping = so.get_mapping()
    blend = json.load(open(BLEND)) if os.path.exists(BLEND) else {"dc_weight": .75, "sup_weight": .25}
    wdc, wsup = blend["dc_weight"], blend["sup_weight"]

    dcp = dc.predict(dc_model, home, away)
    dc_p = np.array([dcp["win_h"], dcp["draw"], dcp["win_a"]])
    form = so.current_form()
    hf, af = form.get(home, 0), form.get(away, 0)
    sup_p = np.array(so.probs_from_rating(mapping, hf - af))
    p = wdc * dc_p + wsup * sup_p
    p = p / p.sum()
    ph, pdw, pa = p

    fav, favp = (home, ph) if ph >= pa else (away, pa)
    shares = ps.load_shares()
    sc_h = ps.match_scorers(home, dcp["xg_h"], shares)
    sc_a = ps.match_scorers(away, dcp["xg_a"], shares)

    rule = "  " + "─" * 44
    print(f"\n  🔮  {home}  vs  {away}")
    print(rule)
    print(f"  ⭐ Favourite: {fav} ({favp*100:.0f}%)")
    print(f"  xG {dcp['xg_h']:.2f} – {dcp['xg_a']:.2f}    recent form {hf:+d} / {af:+d}")
    print(rule)
    print("  " + prob_bar(ph, pdw, pa))
    print(f"  █ {home} {ph*100:.0f}%   ░ Draw {pdw*100:.0f}%   ▒ {away} {pa*100:.0f}%")
    print(f"  fair odds:  {1/ph:.2f} home  /  {1/pdw:.2f} draw  /  {1/pa:.2f} away")
    print(rule)
    scores = "    ".join(f"{i}-{j} {pr*100:.0f}%" for i, j, pr in top_scores(dc_model, home, away))
    print(f"  Most likely scores:  {scores}")
    print(rule)
    for team, sc in ((home, sc_h), (away, sc_a)):
        print(f"  🎯 Anytime scorer — {team}")
        if sc:
            print("     " + "   ".join(f"{pl} {pr*100:.0f}%" for pl, pr in sc))
        else:
            print("     no scorer data (promoted / all-new squad)")
    print(rule)
    print("  Dixon-Coles × supremacy-form blend · squad-based scorers · not betting advice")
    if not dcp["known"]:
        print("  ⚠ a team has no Dixon-Coles rating (promoted/new) — odds unreliable")


if __name__ == "__main__":
    main()
