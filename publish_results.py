"""
Whizard results table -- ROI and CLV by model -> market -> sub-market.

One command, no arguments, safe to call at the end of every model run:

    python3 /mnt/d/Python/whizard-results/publish_results.py

It rebuilds EVERYTHING from the source ledgers each time (models never write into
the JSON themselves, so two models finishing at once cannot collide, and a fix to
a ledger flows through on the next run). Output is deterministic, so git only sees
a change when a number changed; it then commits and pushes to GitHub Pages.

Scope (user, 2026-09-28): NFL + CFB all official markets, MLB = F5 + K Props,
WNBA = mainlines + rebounds. OFFICIAL BETS ONLY.

FROZEN RECORD. The first time a bet shows up as official it is written to
state/archive.csv and stays in the record from then on, even if a later rule
change would no longer call it official (CFB recomputes "official" on every
export). A bet can only ENTER the archive while its game is recent (FREEZE_DAYS),
so loosening a rule later cannot slide old winners into the record. The very
first run backfills everything and marks it reconstructed.

CLV -- one definition for every sport (user, 2026-09-28): the odds we got versus
where the odds closed, with the vig taken out of both.
    clv_prob = p_close - open_fair          (points)
    clv_pct  = p_close / open_fair - 1      (the CLV % shown on the page: how much
                                             better our fair odds were than the close's)
    both are the SAME book's no-vig probability of OUR side at OUR number;
    positive means the market moved toward us. No valid close  =>  NaN, never 0.
There is deliberately no EV-at-close figure: p_close * decimal - 1 carries the vig,
so an unmoved line scores minus the hold, which reads as a loss that is not one.
Where a sport cannot meet that definition the bet gets NaN, not a guess:
    NFL   one-sided rows (anytime TD, alt rungs) have no no-vig fair -> NaN
    WNBA  close read from the per-book snapshot tape; the line moved -> NaN
    K     the line moved -> NaN
    F5    the close is best-line consensus, not per book (as stored)
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PY = Path("/mnt/d/Python")
STATE = ROOT / "state" / "archive.csv"
OUT = ROOT / "results.json"

FREEZE_DAYS = 7          # a new official bet can only enter the record this close to its game
VOID_AFTER_DAYS = 4      # ungraded this long after the game = void (NFL DNP props never settle)
THIN_N = 50              # below this a cell is flagged as too small to read
HEARTBEAT_H = 0.9        # re-stamp `checked` every hourly run, so "Updated" is never >1h old
BOOT = 2000
SEED = 20260928

COLS = ["sport", "model", "market", "sub", "key", "game", "date", "book", "odds", "dec",
        "stake", "result", "pnl_flat", "pnl_kelly", "open_fair", "p_close", "clv_prob", "clv_pct"]


# --------------------------------------------------------------------------- helpers
def dec_from_american(a):
    a = pd.to_numeric(a, errors="coerce")
    return np.where(a > 0, 1 + a / 100.0, 1 + 100.0 / a.abs())


def settle(dec, stake, result):
    """P&L for flat (1u) and kelly stakes from W/L/P/V."""
    win = result.eq("W")
    loss = result.eq("L")
    flat = np.where(win, dec - 1, np.where(loss, -1.0, 0.0))
    kel = np.where(win, stake * (dec - 1), np.where(loss, -stake, 0.0))
    return flat, kel


def clv(p_close, open_fair):
    """No-vig close minus no-vig open, our side. NaN when either is missing."""
    p = pd.to_numeric(pd.Series(p_close), errors="coerce")
    p = p.where((p > 0) & (p < 1))
    return p.to_numpy(), (p - pd.to_numeric(pd.Series(open_fair), errors="coerce").to_numpy()).to_numpy()


def utc(s, naive_tz="UTC"):
    """Parse mixed timestamp strings to UTC. Strings without an offset are read in naive_tz
    (the WNBA ledgers write local Eastern time with no offset beside offset-carrying rows)."""
    s = pd.Series(s).astype(str)
    has_tz = s.str.contains(r"(?:[+-]\d\d:?\d\d|Z)$", regex=True)
    out = pd.to_datetime(s.where(has_tz), utc=True, format="mixed", errors="coerce")
    naive = pd.to_datetime(s.where(~has_tz), format="mixed", errors="coerce")
    naive = naive.dt.tz_localize(naive_tz, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    return out.where(has_tz, naive)


def fav_dog(point_or_price):
    v = pd.to_numeric(point_or_price, errors="coerce")
    return np.where(v < 0, "Favorite", np.where(v > 0, "Underdog", "Pick'em"))


# --------------------------------------------------------------------------- NFL
def load_nfl() -> pd.DataFrame:
    root = PY / "nfl-origination-model"
    live = root / "data/processed/live"
    sched = pd.read_parquet(root / "data/processed/schedules/schedules_2026.parquet")
    sched["kick"] = pd.to_datetime(sched["gameday"] + " " + sched["gametime"]) \
        .dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    kick = sched.set_index("game_id")["kick"]
    out = []
    for fam in ("main", "derivatives", "props"):
        d = pd.read_parquet(live / f"clv_{fam}_ledger.parquet")
        d = d[d["official"].fillna(0).astype(float) == 1].copy()
        if d.empty:
            continue
        d["kick"] = d["game_id"].map(kick)
        d["date"] = d["kick"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
        d["dec"] = d["dec"].astype(float)
        res = d["result"].map({"win": "W", "loss": "L", "push": "P"})
        stale = d["kick"] < pd.Timestamp.now(tz="UTC") - timedelta(days=VOID_AFTER_DAYS)
        d["result"] = res.where(res.notna(), np.where(stale, "V", None))
        d["stake"] = d["stake_official"].astype(float)
        d["key"] = f"nfl|{fam}|" + d[[c for c in ("game_id", "market", "submarket", "player_id",
                                                  "side", "point", "bookmaker")
                                      if c in d.columns]].astype(str).agg("|".join, axis=1)
        side = d["side"].astype(str)
        if fam == "main":
            mk = d["market"].map({"spreads": "Spread", "totals": "Total", "h2h": "Moneyline"})
            sub = np.where(d["market"].eq("totals"), side.str.title(),
                           fav_dog(np.where(d["market"].eq("h2h"), d["price"], d["point"])))
            model = "Mainlines"
            two_sided = pd.Series(True, index=d.index)
        elif fam == "derivatives":
            kind = d["submarket"].str.split(r"[_:]", n=1).str[0]
            per = d["submarket"].str.extract(r"_(h\d|q\d)$")[0].str.upper().fillna("Game")
            mk = kind.map({"spreads": "Spread", "totals": "Total", "team": "Team Total",
                           "h2h": "Moneyline"}).fillna(kind)
            mk = np.where(mk.eq("Team Total"), mk, per + " " + mk)
            sub = np.where(side.str.lower().isin(["over", "under"]), side.str.title(),
                           fav_dog(d["point"]))
            model = "Derivatives"
            two_sided = pd.Series(True, index=d.index)
        else:
            mk = d["submarket"].str.replace("_", " ").str.title() \
                .str.replace("Td", "TD").str.replace("Yds", "Yds")
            alt = d["is_alt"].fillna(0).astype(float).eq(1)
            sub = np.where(side.eq("Yes"), "Yes", np.where(alt, "Alt " + side, side))
            model = "Props"
            two_sided = ~(side.eq("Yes") | alt)
        d["model"], d["market"], d["sub"] = model, mk, sub
        d["game"] = "nfl|" + d["game_id"]
        d["odds"], d["book"] = d["price"], d["bookmaker"]
        d["open_fair"] = d["fair_open"].where(two_sided)
        # a close is the book's last quote within 24h before kickoff
        ct = utc(d["current_time"])
        good = two_sided & (ct < d["kick"]) & (d["kick"] - ct <= timedelta(hours=24))
        d["p_close"], d["clv_prob"] = clv(d["current_fair"].where(good), d["open_fair"])
        d["sport"] = "NFL"
        out.append(d)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=COLS)


# --------------------------------------------------------------------------- CFB
def load_cfb() -> pd.DataFrame:
    root = PY / "ncaaf-origination-model"
    w = pd.DataFrame(json.loads((root / "hub/public/data/wagers.json").read_text())["wagers"])
    w = w[(w["off"] == True) & (w["tracked"] == True)].copy()   # noqa: E712
    g = pd.read_parquet(root / "data/marts/live_graded.parquet")
    g = g.rename(columns={"home_team": "home", "away_team": "away", "bookmaker": "book",
                          "market": "mkt", "week": "wk"})
    g["team"] = g["team"].fillna("")
    w["team"] = w["team"].fillna("")
    on = ["wk", "home", "away", "mkt", "side", "team", "book"]
    d = w.merge(g[on + ["point", "price", "market_prob", "p_close", "stake_kelly", "won"]]
                .rename(columns={"price": "g_price", "won": "g_won"}),
                on=on, how="left")
    d["dec"] = dec_from_american(d["price"])
    d["stake"] = d["stake"].astype(float) * 100          # bankroll fraction -> units
    won = d["won"]
    d["result"] = np.where(won.eq(True), "W", np.where(won.eq(False), "L", None))
    d["date"] = utc(d["ko"]).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    mk = d["mkt"].map({"h2h": "Moneyline", "spreads": "Spread", "totals": "Total",
                       "team_totals": "Team Total"})
    sub = np.where(d["mkt"].isin(["totals", "team_totals"]), d["side"].str.title(),
                   fav_dog(np.where(d["mkt"].eq("h2h"), d["price"], d["pt"])))
    d["sport"], d["model"], d["market"], d["sub"] = "CFB", "CFB", mk, sub
    d["key"] = "cfb|" + d[["gid", "mkt", "side", "team", "book"]].astype(str).agg("|".join, axis=1)
    d["game"] = "cfb|" + d["gid"].astype(str)
    d["odds"] = d["price"]
    d["open_fair"] = d["market_prob"].fillna(d["kp"])
    d["p_close"], d["clv_prob"] = clv(d["p_close"], d["open_fair"])
    return d


# --------------------------------------------------------------------------- WNBA
def _wnba_official(market, edge):
    # mirrors WNBA src/tracking/ledger.py:is_official_pick (stored flag is NaN pre-06-29)
    side = market.str.contains("ml|ats")
    tot = market.isin(["over", "under"])
    return (side & (edge >= 0.17)) | (tot & (edge >= 0.04) & (edge <= 0.10))


def _wnba_close(d, tape_game, tape_prop):
    """The opening book's own last snapshot after the bet opened, same line; NaN otherwise.

    The ledger's current_* is the best price across ALL books (refresh_closing_lines), so
    it cannot give a per-book close. The snapshot tape is one row per book per run, with
    that book's de-vigged mkt_prob, so the close comes from there."""
    out = pd.Series(np.nan, index=d.index)
    for tape, key, mask in ((tape_game, ["game_date", "home_team", "away_team", "market"],
                             d["model"].eq("Mainlines")),
                            (tape_prop, ["game_date", "player_id", "stat", "side"],
                             d["model"].eq("Rebounds"))):
        if not mask.any():
            continue
        t = tape.copy()
        for flag in ("is_alternate", "is_yesno"):
            if flag in t:
                t = t[~t[flag].fillna(False).astype(bool)]
        t["ts"] = utc(t["run_ts"], "America/New_York")
        t = t.sort_values("ts")
        b = d.loc[mask, key + ["bookmaker", "line", "open_time"]].copy()
        b["_i"] = b.index
        b["open_ts"] = utc(b["open_time"], "America/New_York")
        for c in key:
            if c == "player_id":                     # float after the concat with game rows
                t[c], b[c] = (pd.to_numeric(x[c]).astype("Int64") for x in (t, b))
            t[c], b[c] = t[c].astype(str), b[c].astype(str)
        m = b.merge(t[key + ["bookmaker", "ts", "line", "mkt_prob"]], on=key + ["bookmaker"],
                    suffixes=("", "_c"))
        m = m[m["ts"] > m["open_ts"]]
        last = m.sort_values("ts").groupby("_i").tail(1)
        same = (last["line_c"].fillna(-999) - last["line"].fillna(-999)).abs() < 0.25
        out.loc[last.loc[same, "_i"]] = last.loc[same, "mkt_prob"].to_numpy()
    return out


def load_wnba() -> pd.DataFrame:
    root = PY / "WNBA/data/processed"
    g = pd.read_csv(root / "results_ledger.csv")
    g = g[_wnba_official(g["market"], g["edge"])].copy()
    g["model"] = "Mainlines"
    g["market_"] = g["market"].map({"home_ml": "Moneyline", "away_ml": "Moneyline",
                                    "home_ats": "Spread", "away_ats": "Spread",
                                    "over": "Total", "under": "Total"})
    own_pt = np.where(g["market"].eq("away_ats"), -g["line"], g["line"])  # line is the HOME spread
    g["sub"] = np.where(g["market"].isin(["over", "under"]), g["market"].str.title(),
                        fav_dog(np.where(g["market"].str.endswith("ml"), g["price"], own_pt)))
    g["key"] = "wnba|g|" + g[["game_date", "home_team", "away_team", "market"]].agg("|".join, axis=1)

    p = pd.read_csv(root / "results_ledger_props.csv")
    p = p[(p["stat"] == "rebounds") & (p["edge"] >= 0.10)].copy()
    p["model"], p["market_"], p["sub"] = "Rebounds", "Rebounds", p["side"].str.title()
    p["key"] = "wnba|reb|" + p[["game_date", "player_id", "stat", "side"]].astype(str) \
        .agg("|".join, axis=1)

    d = pd.concat([g, p], ignore_index=True)
    p_close = _wnba_close(d, pd.read_parquet(root / "snapshots_game.parquet"),
                          pd.read_parquet(root / "snapshots_props.parquet"))
    d["market"] = d.pop("market_")
    d["sport"], d["date"], d["book"], d["odds"] = "WNBA", d["game_date"], d["bookmaker"], d["price"]
    d["game"] = "wnba|" + d["game_date"] + "|" + d["home_team"] + "|" + d["away_team"]
    d["dec"] = dec_from_american(d["price"])
    d["stake"] = d["kelly"].astype(float)
    d["result"] = np.where(d["bet_won"].eq(1), "W", np.where(d["bet_won"].eq(0), "L", None))
    graded = d["flat_profit"].notna() & d["bet_won"].isna()
    d.loc[graded, "result"] = np.where(d.loc[graded, "model"].eq("Rebounds"), "V", "P")
    d["open_fair"] = d["mkt_prob"]
    d["p_close"], d["clv_prob"] = clv(p_close, d["open_fair"])
    return d


# --------------------------------------------------------------------------- MLB
def load_f5() -> pd.DataFrame:
    d = pd.read_csv(PY / "MLB/data/processed/nrfi/f5_results_ledger.csv")
    d = d[(d["bet"] == 1) & (d["result"] != "no_data")].copy()
    d = d.sort_values("game_date").drop_duplicates(["game_pk", "market"], keep="last")
    own_pt = np.where(d["pick"].eq("home"), d["line"], -d["line"])       # line is the HOME line
    d["sport"], d["model"], d["market"], d["sub"] = "MLB", "F5", "Run Line", fav_dog(own_pt)
    d["date"], d["book"], d["odds"] = d["game_date"], d["shop_book"], d["price"]
    d["key"] = "f5|" + d["game_pk"].astype(str) + "|" + d["market"]
    d["game"] = "mlb|" + d["game_pk"].astype(str)
    d["dec"] = dec_from_american(d["price"])
    d["stake"] = d["kelly_stake"].astype(float)
    d["result"] = d["result"].map({"win": "W", "loss": "L", "push": "P"})
    d["open_fair"] = d["fair_prob"]
    refreshed = utc(d["current_time"]) > utc(d["open_time"])
    d["p_close"], d["clv_prob"] = clv(d["current_fair_prob"].where(refreshed), d["open_fair"])
    return d


def load_k() -> pd.DataFrame:
    d = pd.read_csv(PY / "MLB/data/processed/k_props/results_ledger.csv")
    d = d[d["dk_bet"] == 1].copy()
    over = d["best_side"].eq("over")
    d["sport"], d["model"], d["market"] = "MLB", "K Props", "Strikeouts"
    d["sub"] = d["best_side"].str.title()
    d["date"], d["book"] = d["game_date"], d["shop_book"]
    d["key"] = "k|" + d["game_date"] + "|" + d["pitcher_id"].astype(str)
    d["game"] = "mlb|" + d["game_date"] + "|" + d[["team", "opp"]].apply(sorted, axis=1).str.join("-")
    # P&L is paid at the shop price; stake and CLV live at the primary book
    d["odds"] = np.where(over, d["shop_over"], d["shop_under"])
    d["dec"] = dec_from_american(pd.Series(d["odds"], index=d.index))
    d["stake"] = d["kelly_stake"].astype(float)
    res = d["result"].astype(str)
    d["result"] = np.where(res.eq("push"), "P",
                  np.where(res.isin(["void", "no_line"]), "V",
                  np.where(d["bet_won"].eq(1), "W", np.where(d["bet_won"].eq(0), "L", None))))
    io, iu = 1 / dec_from_american(d["over_price"]), 1 / dec_from_american(d["under_price"])
    co, cu = 1 / dec_from_american(d["current_over"]), 1 / dec_from_american(d["current_under"])
    d["open_fair"] = np.where(over, io / (io + iu), iu / (io + iu))
    close = pd.Series(np.where(over, co / (co + cu), cu / (co + cu)), index=d.index)
    good = (utc(d["current_time"]) > utc(d["open_time"])) & \
           ((d["current_line"] - d["line"]).abs() < 0.25)
    d["p_close"], d["clv_prob"] = clv(close.where(good), d["open_fair"])
    return d


LOADERS = {"NFL": load_nfl, "CFB": load_cfb, "WNBA": load_wnba, "F5": load_f5, "K": load_k}


# --------------------------------------------------------------------------- record
def build_record() -> tuple[pd.DataFrame, list[str]]:
    frames, errors = [], []
    for name, fn in LOADERS.items():
        try:
            d = fn()
            d = d.drop_duplicates("key", keep="last")
            d["clv_pct"] = np.nan
            flat, kel = settle(d["dec"].astype(float), d["stake"].astype(float),
                               d["result"].astype(object))
            d["pnl_flat"], d["pnl_kelly"] = flat, kel
            frames.append(d[COLS])
        except Exception as e:                       # one broken ledger must not blank the page
            errors.append(f"{name}: {type(e).__name__}: {e}")
    live = pd.concat(frames, ignore_index=True)
    live["reconstructed"] = False

    today = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
    if STATE.exists():
        arch = pd.read_csv(STATE, dtype={"key": str})
        arch = arch.drop(columns=["clv_ev"], errors="ignore")   # retired: carried the vig
        known = set(arch["key"])
        fresh = ~live["key"].isin(known)
        recent = pd.to_datetime(live["date"]) >= today - timedelta(days=FREEZE_DAYS)
        live = live[~fresh | recent]
        # frozen bets whose loader failed, or that a rule change dropped, keep their last snapshot
        kept = arch[~arch["key"].isin(live["key"])]
        prior = arch.set_index("key")["reconstructed"]
        live["reconstructed"] = live["key"].map(prior).fillna(False).astype(bool)
        record = pd.concat([live, kept], ignore_index=True)
    else:
        live["reconstructed"] = True                 # first run: backfilled from today's ledgers
        record = live
    record = record.sort_values(["sport", "model", "date", "key"]).reset_index(drop=True)
    # CLV %, recomputed for every row so archived snapshots carry it too
    of = pd.to_numeric(record["open_fair"], errors="coerce")
    record["clv_pct"] = pd.to_numeric(record["p_close"], errors="coerce") / of.where(of > 0) - 1
    STATE.parent.mkdir(exist_ok=True)
    record.to_csv(STATE, index=False, float_format="%.6g")
    return record, errors


# --------------------------------------------------------------------------- stats
def _ci(num_g, den_g, rng):
    """95% cluster-bootstrap CI of sum(num)/sum(den), resampling whole games."""
    G = len(num_g)
    if G < 2 or den_g.sum() <= 0:
        return None
    idx = rng.integers(0, G, size=(BOOT, G))
    den = den_g[idx].sum(1)
    r = np.where(den > 0, num_g[idx].sum(1) / np.where(den > 0, den, 1), np.nan)
    lo, hi = np.nanpercentile(r, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def stats(d: pd.DataFrame, rng) -> dict:
    dec = d[d["result"].isin(["W", "L"])]
    s = {"n": int(len(dec)), "w": int(d["result"].eq("W").sum()), "l": int(d["result"].eq("L").sum()),
         "p": int(d["result"].eq("P").sum()), "v": int(d["result"].eq("V").sum()),
         "pending": int(d["result"].isna().sum())}
    if dec.empty:
        return s
    by = dec.groupby("game")
    stake_g = by["stake"].sum().to_numpy(float)
    kel_g = by["pnl_kelly"].sum().to_numpy(float)
    flat_g = by["pnl_flat"].sum().to_numpy(float)
    cnt_g = by.size().to_numpy(float)
    staked = float(stake_g.sum())
    s.update({
        "win_pct": round(s["w"] / s["n"], 4),
        "staked": round(staked, 2), "avg_stake": round(staked / s["n"], 3),
        "units": round(float(kel_g.sum()), 2),
        "roi": round(float(kel_g.sum()) / staked, 4) if staked > 0 else None,
        "roi_ci": _ci(kel_g, stake_g, rng),
        "flat_units": round(float(flat_g.sum()), 2),
        "flat_roi": round(float(flat_g.sum()) / s["n"], 4),
        "flat_roi_ci": _ci(flat_g, cnt_g, rng),
        "thin": s["n"] < THIN_N,
    })
    c = d[d["clv_prob"].notna()]
    s["clv_n"] = int(len(c))
    if len(c):
        cg = c.groupby("game")
        s["clv_line"] = round(float(c["clv_prob"].mean()), 4)   # no-vig points moved toward us
        s["clv_pct"] = round(float(c["clv_pct"].mean()), 4)     # headline CLV %
        s["clv_ci"] = _ci(cg["clv_prob"].sum().to_numpy(float), cg.size().to_numpy(float), rng)
        moved = c[c["clv_prob"].abs() > 1e-9]
        s["clv_moved"] = int(len(moved))
        s["beat_close"] = round(float((moved["clv_prob"] > 0).mean()), 4) if len(moved) else None
    return s


def summarize(rec: pd.DataFrame) -> list[dict]:
    rng = np.random.default_rng(SEED)                 # fixed seed: same data -> same JSON
    rows = []
    levels = [["sport"], ["sport", "model"], ["sport", "model", "market"],
              ["sport", "model", "market", "sub"]]
    for lv in levels:
        for k, grp in rec.groupby(lv, sort=True):
            k = k if isinstance(k, tuple) else (k,)
            row = dict(zip(["sport", "model", "market", "sub"], [None] * 4))
            row.update(dict(zip(lv, k)))
            row["level"] = len(lv)
            row["first"], row["last"] = grp["date"].min(), grp["date"].max()
            row.update(stats(grp, rng))
            rows.append(row)
    return rows


# --------------------------------------------------------------------------- publish
def bets_payload(rec):
    r = lambda v, n=4: None if pd.isna(v) else round(float(v), n)   # noqa: E731
    return [{"s": x.sport, "m": x.model, "k": x.market, "u": x.sub, "d": x.date, "g": x.game,
             "o": None if pd.isna(x.odds) else int(float(x.odds)), "st": r(x.stake, 3),
             "r": None if pd.isna(x.result) else x.result, "pf": r(x.pnl_flat), "pk": r(x.pnl_kelly),
             "cp": r(x.clv_prob), "cr": r(x.clv_pct)} for x in rec.itertuples()]


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)


def main():
    rec, errors = build_record()
    body = {"groups": summarize(rec), "bets": bets_payload(rec), "errors": errors,
            "method": {"roi": "kelly units won / kelly units staked; pushes and voids excluded",
                       "ci": f"95% bootstrap, resampling whole games ({BOOT} draws)",
                       "clv": "no-vig close minus no-vig open for our side, same book, same number; "
                              "blank when no close was captured",
                       "thin_n": THIN_N}}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:16]
    old = json.loads(OUT.read_text()) if OUT.exists() else {}
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    if old.get("hash") == digest:
        # Heartbeat. Nothing new was graded, but the page still needs to know
        # the job is alive, or "no new results" and "the job stopped" look the
        # same. Re-stamp `checked` at most every HEARTBEAT_H hours so the repo
        # does not take a commit every hour.
        last = old.get("checked") or old.get("generated") or ""
        try:
            age_h = (now - datetime.strptime(last, "%Y-%m-%dT%H:%M:%SZ")
                     .replace(tzinfo=timezone.utc)).total_seconds() / 3600
        except ValueError:
            age_h = 1e9
        # --heartbeat forces one now: an end-to-end test that touches no results
        if (age_h < HEARTBEAT_H and "--heartbeat" not in sys.argv) or "--no-push" in sys.argv:
            print("results unchanged")
            return 0
        old["checked"] = stamp
        OUT.write_text(json.dumps(old, separators=(",", ":"), default=str))
        git("add", "-A")
        git("commit", "-m", f"heartbeat {stamp}")
        p = git("push", "-q")
        print("heartbeat pushed" if p.returncode == 0 else f"push failed: {p.stderr.strip()}")
        return 0
    # A dry run must not record the hash, or the next real run would see
    # "unchanged" and never push what the dry run computed.
    dry = "--no-push" in sys.argv
    body = {"generated": stamp, "checked": stamp, "hash": "" if dry else digest, **body}
    OUT.write_text(json.dumps(body, separators=(",", ":"), default=str))
    tops = [g for g in body["groups"] if g["level"] == 2]
    for g in tops:
        print(f"{g['sport']:5} {g['model']:12} n={g['n']:4}  {g['w']}-{g['l']}  "
              f"roi={g.get('roi')}  clv_n={g.get('clv_n', 0)}  pending={g['pending']}")
    for e in errors:
        print("LOADER FAILED", e, file=sys.stderr)
    if dry:
        return 0
    git("add", "-A")
    git("commit", "-m", f"results {body['generated']}")
    p = git("push", "-q")
    print("pushed" if p.returncode == 0 else f"push failed: {p.stderr.strip()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
