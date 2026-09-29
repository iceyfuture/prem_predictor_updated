"""
Anytime-goalscorer model for the Premier League — the same approach as the World Cup
scorer predictor, adapted to club football and the confirmed 2026/27 squads.

  * A player's scoring RATE is a gamma-Poisson posterior: his recency-weighted Premier
    League goals, shrunk toward an empirical prior for his position.
  * His WEIGHT is that rate times his expected share of the minutes.
  * His SHARE of the club is weight / sum(weight) over the squad. Because the pool is the
    actual current squad, transfers are handled correctly -- a striker's past goals move
    with him to his new club.
  * For a fixture with team expected goals lambda (from Dixon-Coles), a player's goals in
    the match ~ Poisson(share * lambda), so P(scores anytime) = 1 - exp(-THETA*share*lam).

RULE 48 (audit §7). Until 2026-09-22 the share was simply `w_goals / sum(w_goals)`, and
`load_shares` dropped everyone with `w_goals <= 0`. That gave 287 of 614 squad players
(46.7%) -- including 22 forwards and 129 players with no Premier League minutes at all --
a probability of exactly zero. Over the two graded gameweeks, 14 of 47 players who
actually scored had been assigned 0%. A model that says an event is impossible, and is
then wrong 14 times in two weeks, is not merely mis-tuned; log loss on those rows is
infinite. The prior removes the impossibility. It is NOT an accuracy claim -- see
MODEL_RULES Rule 48 for the held-out numbers, which do not establish one.

Constants are empirical or principled, never tuned against the outcomes they are scored on:
  PRIOR_G90    goals per 90 in a player's FIRST Premier League season, measured over
               2016-17..2025-26 on players with >=270 minutes (premier_league_history).
  NEW_REL_MIN  median in-club relative minutes of players who DO have history, by position,
               used as the minutes expectation for a player who has none.
  PRIOR_K      10 ninety-minute appearances before observation outweighs the prior. A
               round, defensible shrinkage scale, not fitted.

THETA is a playing-time discount (1.0 = shares already encode minutes; lower it if a
scorer backtest later shows over-prediction).
"""
import os
import re
import unicodedata
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
def _find(*rel, default=None):
    bases = [HERE, os.path.dirname(HERE)]
    for b in bases:
        for r in rel:
            q = os.path.join(b, r)
            if os.path.exists(q):
                return q
    return os.path.join(HERE, default or rel[0])


LINKED = _find("outputs/squad_2026_27_linked.csv", "data/squad_2026_27_linked.csv",
               "../data/squad_2026_27_linked.csv", default="outputs/squad_2026_27_linked.csv")
THETA = 1.0
TOP_N = 5

# goals per 90 in a player's first Premier League season (>=270 min, 2016-17..2025-26).
# GKP is the all-season rate: 1 goal in 683,955 minutes. Small, but not impossible.
PRIOR_G90 = {"FWD": 0.3838, "MID": 0.1432, "DEF": 0.0468, "GKP": 0.000132}
# median relative in-club minutes among players WITH history, by position
NEW_REL_MIN = {"FWD": 0.280, "MID": 0.355, "DEF": 0.415, "GKP": 0.133}
PRIOR_K = 10.0
DEFAULT_POS = "MID"

# The dashboard and the ledger used different club names for the same five clubs, so a
# quarter of graded rows never joined. Canonical form is the squad file's.
TEAM_ALIAS = {"Coventry City": "Coventry", "Hull City": "Hull", "Ipswich Town": "Ipswich",
              "Man Utd": "Man United", "Spurs": "Tottenham",
              "Tottenham Hotspur": "Tottenham", "Manchester United": "Man United",
              "Manchester City": "Man City", "Newcastle United": "Newcastle",
              "Nottingham Forest": "Nott'm Forest", "Wolverhampton Wanderers": "Wolves"}


def canon_team(name):
    """Canonical club name. Unknown names pass through unchanged."""
    return TEAM_ALIAS.get((name or "").strip(), (name or "").strip())


# FPL says GK where the squad file says GKP; everything else already agrees.
_POS_ALIAS = {"GK": "GKP", "GKP": "GKP", "DEF": "DEF", "MID": "MID", "FWD": "FWD"}


def canon_pos(pos):
    """Canonical position code, or '' when it is missing or unrecognised.

    The squad file carries NaN for a few players, so this must survive a float.
    """
    if not isinstance(pos, str):
        return ""
    return _POS_ALIAS.get(pos.strip().upper(), "")


def _surname(name):
    n = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode()
    n = re.sub(r"[.\-']", " ", n).lower()
    toks = [t for t in n.split() if t not in
            ("de", "van", "dos", "da", "der", "den", "el", "al", "di")]
    return toks[-1] if toks else ""


def player_rate(w_goals, w_minutes, position):
    """Gamma-Poisson posterior mean goals per 90.

    A player with no Premier League minutes falls back to the whole prior; one with a long
    record is dominated by what he actually did. Continuous in between -- there is no
    threshold at which a player stops being a newcomer.
    """
    pos = position if position in PRIOR_G90 else DEFAULT_POS
    n90 = max(float(w_minutes or 0.0), 0.0) / 90.0
    return (max(float(w_goals or 0.0), 0.0) + PRIOR_K * PRIOR_G90[pos]) / (n90 + PRIOR_K)


_POSITIONS = None


def load_positions():
    """{(club, surname): position} for the squad file, but ONLY where that surname is
    unique within the club.

    An ambiguous surname is deliberately absent rather than resolved to one of the
    candidates: Chelsea have both Wesley Fofana (DEF) and David Datro Fofana (FWD), and
    picking either would be a guess.
    """
    global _POSITIONS
    if _POSITIONS is not None:
        return _POSITIONS
    df = pd.read_csv(LINKED)
    if "position" not in df.columns:
        df["position"] = DEFAULT_POS
    seen, dupes = {}, set()
    for team, player, pos in zip(df.team_2026_27, df.player, df.position):
        k = (canon_team(team), _surname(player))
        if k in seen:
            dupes.add(k)                 # ANY repeat, even at the same position
        seen[k] = canon_pos(pos)
    _POSITIONS = {k: v for k, v in seen.items() if k not in dupes}
    return _POSITIONS


def player_pos(team, player):
    """This player's canonical position, from the squad file row itself."""
    return canon_pos(_POS_BY_NAME.get((canon_team(team), player), ""))


_POS_BY_NAME = {}


def load_shares():
    """team -> Series(player -> share of the club's goals), summing to 1.

    Every player in the squad gets a share. None is zero, because none of them is
    incapable of scoring.
    """
    df = pd.read_csv(LINKED)
    df["w_goals"] = pd.to_numeric(df.get("w_goals"), errors="coerce").fillna(0.0)
    df["w_minutes"] = pd.to_numeric(df.get("w_minutes"), errors="coerce").fillna(0.0)
    if "position" not in df.columns:
        df["position"] = DEFAULT_POS
    shares = {}
    for team, g in df.groupby("team_2026_27"):
        w = {}
        for _, r in g.iterrows():
            pos = r.position if r.position in PRIOR_G90 else DEFAULT_POS
            rate = player_rate(r.w_goals, r.w_minutes, pos)
            # Position-typical minutes, NOT the player's own history. Scaling by his past
            # PL minutes looks reasonable and is badly wrong: it punishes precisely the
            # players the prior exists to rescue. Hull's Oliver McBurnie -- a proven
            # forward, but only 197 weighted PL minutes after a spell outside the league --
            # ranked BELOW a team-mate who has never played in the Premier League at all,
            # because having a little data scored worse than having none. An estimator must
            # never do that. Real expected minutes, where production has them, are passed
            # to scorer_probs() instead; this default is only the no-information case.
            w[r.player] = rate * NEW_REL_MIN[pos]
            _POS_BY_NAME[(canon_team(team), r.player)] = pos
        tot = sum(w.values())
        if tot <= 0:
            continue
        s = pd.Series({pl: v / tot for pl, v in w.items()}).sort_values(ascending=False)
        shares[canon_team(team)] = s
    return shares


def _minutes_for(minutes, team, player):
    """This player's expected minutes, or None if we cannot attribute any to HIM.

    Matching on bare surname is not safe. Chelsea's squad holds Wesley Fofana (DEF) and
    David Datro Fofana (FWD); FPL lists exactly one Fofana at Chelsea, a defender. Keyed by
    surname alone both of them resolve to the defender's minutes, so David Datro Fofana --
    who has not been at the club for years -- inherited a real player's minutes, cleared
    the "is he playing" check, and was published at 26% to score. Hull's goalkeeper Dillon
    Phillips reached Man City's Kalvin Phillips the same way.

    So: position first, and a bare surname only when it is unambiguous on BOTH sides. When
    the surname is shared and the positions do not separate them, return None -- the caller
    treats that as "not playing", which is the safe direction for a name we cannot resolve.
    """
    if minutes is None:
        return None
    sn = _surname(player)
    pos = player_pos(team, player)
    if pos:
        v = minutes.get((team, sn, pos))
        if v is not None:
            return v
    # The bare-surname key exists when the surname is unique on the FEED's side. It must
    # also be unique on OURS, or the wrong namesake collects it: FPL has one Fofana at
    # Chelsea, our squad file has two, and without this check the forward still inherited
    # the defender's minutes.
    if (team, sn) not in load_positions():
        return None
    return minutes.get((team, sn))


def scorer_probs(team, lam, shares, avail=None, minutes=None):
    """{player: P(scores anytime)} for a club's whole squad.

    THE single source of truth for this number. The fixture card, the FPL projection and
    the forward-test ledger all read it, so the figure that is displayed is by construction
    the figure that is graded. Before Rule 48 there were three separate calculations and
    the ledger was scoring a pipeline nobody could see.

    `avail` maps (team, surname) -> availability in [0,1] from FPL status/chance-of-playing:
    0 = out/suspended, chance/100 = doubt, 1 = fit. `minutes` optionally maps the same key
    to expected minutes and takes precedence. Unavailable players are removed and their
    share is redistributed over whoever is expected to play, so team expected goals still
    sum to lam.
    """
    team = canon_team(team)
    if team not in shares:
        return {}
    items = list(shares[team].items())
    # Does the minutes map cover this club at all? If the map was built but this club has no
    # entries in it (a feed hiccup, a club FPL has not published), the map is uninformative
    # HERE and absence from it means nothing. If the club IS covered, absence is evidence.
    club_covered = minutes is not None and any(
        _minutes_for(minutes, team, pl) is not None for pl, _ in items)
    adj = []
    for pl, sh in items:
        k = (team, _surname(pl))
        _m = _minutes_for(minutes, team, pl) if minutes is not None else None
        if _m is not None:
            m = max(float(_m), 0.0) / 90.0
        elif club_covered:
            # FPL lists every player registered to a club's squad. Someone in our squad file
            # who is NOT in that list is out on loan, sold, or unregistered -- he is not
            # playing. Falling through to `avail`'s default of 1.0 treated him as a nailed-on
            # starter instead, which is the strongest possible weight and exactly backwards.
            #
            # This was invisible before RULE 48: a player with no Premier League goals had
            # share 0 and could never be listed. Giving everyone a prior made 103 such
            # players eligible, and the 1.0 default then ranked several of them top-three.
            # Brighton's Mark O'Mahony -- 7.6 weighted PL minutes, not in FPL's squad at all
            # -- was published as the club's most likely scorer at 20%.
            m = 0.0
        elif avail:
            m = float(avail.get(k, 1.0))
        else:
            m = 1.0
        adj.append((pl, sh * m))
    tot = sum(w for _, w in adj)
    if tot > 0:
        adj = [(pl, w / tot) for pl, w in adj]
    else:
        adj = items
    return {pl: float(1.0 - np.exp(-THETA * sh * lam)) for pl, sh in adj}


def match_scorers(team, lam, shares, avail=None, minutes=None, n=TOP_N):
    """The top `n` anytime-scorer probabilities for a club, highest first."""
    p = scorer_probs(team, lam, shares, avail=avail, minutes=minutes)
    return sorted(p.items(), key=lambda x: -x[1])[:n]


if __name__ == "__main__":
    import sys
    shares = load_shares()
    if len(sys.argv) == 2:
        team = canon_team(sys.argv[1])
        print(f"{team} scorer shares:")
        for pl, sh in shares.get(team, pd.Series(dtype=float)).head(12).items():
            print(f"  {sh*100:5.1f}%  {pl}")
    else:
        print("teams with scorer data:", ", ".join(sorted(shares)))
