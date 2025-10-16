#!/usr/bin/env python3
"""
FastAPI backend for Live NFL Predictor UI
- Exposes GET /state?event=<ESPN_EVENT_ID>
- Returns JSON with teams, score, predictions (home/away/total), EPA summaries, and recent plays.

Prereqs:
  pip install fastapi uvicorn joblib requests numpy pandas python-dateutil
  # If you plan to retrain: pip install nflreadpy polars statsmodels scikit-learn

Requires EP & GLM artifacts you trained earlier:
  - ep_model.joblib (from your ep_live_v3 training)
  - live_glm.json   (from your GLM training)

Run:
  uvicorn server:app --reload --port 8000
"""

from __future__ import annotations
import json, math, os, time, datetime as dt
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
from dateutil import parser as dparser

# ---------- Config ----------
EP_MODEL_PATH = os.environ.get("EP_MODEL_PATH", "ep_model.joblib")
GLM_JSON_PATH = os.environ.get("GLM_JSON_PATH", "live_glm.json")
POLL_LIMIT = 500   # ESPN plays limit per call

# ---------- EP Approximator (per-down models + isotonic) ----------
EP_FEATS = [
    "yardline_100","ydstogo","log_ydstogo","half_seconds_remaining",
    "posteam_timeouts_remaining","defteam_timeouts_remaining","is_home_offense",
    "goal_to_go","roof_outdoors"
]

def _rehydrate_iso(payload: Dict[str, Any]):
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.X_thresholds_ = np.asarray(payload["x"], dtype=float)
    iso.y_thresholds_ = np.asarray(payload["y"], dtype=float)
    iso.f_min_ = float(iso.y_thresholds_[0])
    iso.f_max_ = float(iso.y_thresholds_[-1])
    return iso

class EPApproximator:
    def __init__(self, path: str):
        b = load(path)
        self.models = {int(k): v for k, v in b["perdown_models"].items()}
        self.calibrators = {int(k): _rehydrate_iso(v) for k, v in b["perdown_calibrators"].items()}
        self.features = b["features"]

    def ep_from_state(self, state: Dict[str, float], down: int) -> float:
        d = int(down) if 1 <= int(down) <= 4 else 1
        mdl = self.models.get(d) or self.models.get(1)
        iso = self.calibrators.get(d)
        feats = {f: float(state.get(f, 0.0)) for f in EP_FEATS}
        feats["log_ydstogo"] = math.log1p(max(0.0, feats.get("ydstogo", 0.0)))
        X = pd.DataFrame([feats])[EP_FEATS].astype(float)
        pred = mdl.predict(X)[0]
        return float(iso.transform([pred])[0] if iso else pred)

    def epa_from_states(self,
                        pre: Dict[str, float], pre_down: int,
                        post: Dict[str, float], post_down: int,
                        possession_changed: bool,
                        offense_scored_points: int = 0,
                        defense_scored_points: int = 0) -> Tuple[float,float,float]:
        ep_before = self.ep_from_state(pre, pre_down)
        ep_after_new = self.ep_from_state(post, post_down)
        ep_after_for_start_off = -ep_after_new if possession_changed else ep_after_new
        net_points = int(offense_scored_points) - int(defense_scored_points)
        epa = (ep_after_for_start_off + net_points) - ep_before
        return ep_before, ep_after_new, epa

# ---------- ESPN Helpers ----------
def _get_json(url: str) -> Dict[str, Any]:
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()

def _team_ctx(event: str):
    j = _get_json(f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={event}")
    comp = j["header"]["competitions"][0]
    home = next(c for c in comp["competitors"] if c["homeAway"] == "home")
    away = next(c for c in comp["competitors"] if c["homeAway"] == "away")
    return {
        "home_id": home["id"], "away_id": away["id"],
        "home_abbr": home["team"]["abbreviation"], "away_abbr": away["team"]["abbreviation"]
    }

def _parse_clock(clock_disp: str) -> int:
    if ":" not in (clock_disp or ""): return 0
    m, s = clock_disp.split(":"); return int(m)*60 + int(s)

def _half_secs(period: int, clock_secs: int) -> int:
    if period in (1,2): return clock_secs + (900 if period == 1 else 0)
    if period in (3,4): return clock_secs + (900 if period == 3 else 0)
    return 0

def _plays(event: str) -> List[Dict[str, Any]]:
    root = _get_json(f"https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/{event}/competitions/{event}/plays?limit={POLL_LIMIT}")
    items = root.get("items", [])
    if items and isinstance(items[0], dict) and "type" in items[0]:
        return items
    return [_get_json(it["$ref"]) for it in items]

def _timeouts_or_default(event: str, offense_is_home: bool) -> Tuple[int,int]:
    try:
        j = _get_json(f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={event}")
        comp = j["header"]["competitions"][0]
        home = next(c for c in comp["competitors"] if c["homeAway"] == "home")
        away = next(c for c in comp["competitors"] if c["homeAway"] == "away")
        home_tos = int(home.get("timeouts", 2)); away_tos = int(away.get("timeouts", 2))
        return (home_tos, away_tos) if offense_is_home else (away_tos, home_tos)
    except Exception:
        return (2,2)

def _roof_flag_from_summary(event: str) -> int:
    try:
        j = _get_json(f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={event}")
        comp = j["header"]["competitions"][0]
        ind = comp.get("venue", {}).get("indoor", None)
        if isinstance(ind, bool): return 0 if ind else 1
    except Exception:
        pass
    return 1

def _state_from_espn(start: Dict[str, Any], period: int, offense_is_home: bool, event: str, roof_out: int):
    down = int(start.get("down", 0) or 0)
    ytg = float(start.get("distance", 0) or 0) or 10.0
    y100 = start.get("yardsToEndzone", None)
    y100 = float(y100 if y100 is not None else 75.0)
    clk = _parse_clock(start.get("clock", {}).get("displayValue", "0:00"))
    hrem = _half_secs(int(period), clk)
    off_tos, def_tos = _timeouts_or_default(event, offense_is_home)
    state = {
        "yardline_100": y100,
        "ydstogo": float(ytg),
        "log_ydstogo": math.log1p(max(0.0, ytg)),
        "half_seconds_remaining": float(hrem),
        "posteam_timeouts_remaining": float(off_tos),
        "defteam_timeouts_remaining": float(def_tos),
        "is_home_offense": 1.0 if offense_is_home else 0.0,
        "goal_to_go": 1.0 if abs(ytg - y100) <= 1.0 else 0.0,
        "roof_outdoors": float(roof_out),
    }
    return state, (down if 1 <= down <= 4 else 1)

def _is_pass(play: Dict[str, Any]) -> Optional[bool]:
    ty = (play.get("type", {}) or {}).get("text", "") or ""
    tx = (play.get("text") or "")
    s = f"{ty} {tx}".lower()
    if any(k in s for k in ["pass", "sacked", "incomplete", "scramble"]): return True
    if any(k in s for k in ["rush", "left guard", "right end", "up the middle", "tackle", "left end"]): return False
    return None

# ---------- GLM loader + helpers ----------
with open(GLM_JSON_PATH, "r") as f:
    GLM = json.load(f)
EP = EPApproximator(EP_MODEL_PATH)

NAMES_H = GLM["home"]["features"]; MU_H = np.asarray(GLM["home"]["mu"]); SD_H = np.asarray(GLM["home"]["sd"]); BETA_H = np.asarray(GLM["home"]["beta"])
NAMES_A = GLM["away"]["features"]; MU_A = np.asarray(GLM["away"]["mu"]); SD_A = np.asarray(GLM["away"]["sd"]); BETA_A = np.asarray(GLM["away"]["beta"])

def _zscore_with(mu: np.ndarray, sd: np.ndarray, x: np.ndarray) -> np.ndarray:
    sd = sd.copy(); sd[sd < 1e-12] = 1.0
    return (x - mu) / sd

def _predict_remaining(beta: np.ndarray, feats: np.ndarray, exposure: float) -> float:
    X1 = np.hstack([1.0, feats])
    lam = float(np.exp(np.log(max(1.0, exposure)) + X1 @ beta))
    return lam

# ---------- FastAPI app ----------
app = FastAPI(title="Live NFL Predictor API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

@app.get("/state")
def state(event: str = Query(..., description="ESPN event/competition id")):
    try:
        teams = _team_ctx(event)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not get event teams: {e}")

    roof_flag = _roof_flag_from_summary(event)
    plays = _plays(event)

    # rolling EPA + pass/rush breakdowns
    agg = {
        teams["home_abbr"]: {"pass_plays":0, "rush_plays":0, "pass_epa_sum":0.0, "rush_epa_sum":0.0},
        teams["away_abbr"]: {"pass_plays":0, "rush_plays":0, "pass_epa_sum":0.0, "rush_epa_sum":0.0},
    }

    recent = []
    home_pts = 0; away_pts = 0

    for p in reversed(plays):  # oldest->newest; we'll collect recents later
        pid = p.get("id")
        period = int(p.get("period", {}).get("number", 0) or 0)
        st, en = p.get("start", {}), p.get("end", {})
        # poss at start
        poss_ref = st.get("team", {}).get("$ref") or st.get("possession", {}).get("$ref")
        poss_id = poss_ref.split("/")[-1] if isinstance(poss_ref, str) else None
        offense_is_home = (poss_id == teams["home_id"])
        post_ref = en.get("team", {}).get("$ref") or en.get("possession", {}).get("$ref")
        post_id = post_ref.split("/")[-1] if isinstance(post_ref, str) else poss_id
        changed = (post_id != poss_id)

        pre, pre_down = _state_from_espn(st, period, offense_is_home, event, roof_flag)
        post, post_down = _state_from_espn(en, period, (post_id == teams["home_id"]), event, roof_flag)

        off_pts = int(p.get("scoreValue", 0) or 0) if not changed else 0
        def_pts = int(p.get("scoreValue", 0) or 0) if changed else 0

        epb, epaft, epa = EP.epa_from_states(pre, pre_down, post, post_down, changed, off_pts, def_pts)

        # score if available
        if en.get("homeScore") is not None: home_pts = int(en.get("homeScore"))
        if en.get("awayScore") is not None: away_pts = int(en.get("awayScore"))

        # pass/rush classify
        ispass = _is_pass(p)
        off_abbr = teams["home_abbr"] if offense_is_home else teams["away_abbr"]
        if ispass is True:
            agg[off_abbr]["pass_plays"] += 1
            agg[off_abbr]["pass_epa_sum"] += epa
        elif ispass is False:
            agg[off_abbr]["rush_plays"] += 1
            agg[off_abbr]["rush_epa_sum"] += epa

        if len(recent) < 40:
            recent.append({
                "id": pid, "q": period, "down": st.get("down","?"), "dist": st.get("distance","?"),
                "epb": float(epb), "epa": float(epa),
                "desc": p.get("text") or p.get("type", {}).get("text") or ""
            })

    # features at NOW
    def mk(team_abbr: str):
        d = agg[team_abbr]
        pass_epa = (d["pass_epa_sum"] / d["pass_plays"]) if d["pass_plays"]>0 else 0.0
        rush_epa = (d["rush_epa_sum"] / d["rush_plays"]) if d["rush_plays"]>0 else 0.0
        pass_sr  = (d["pass_epa_sum"] > 0) * 0.0  # placeholder (we don't count successes separate server-side)
        rush_sr  = (d["rush_epa_sum"] > 0) * 0.0
        # We’ll estimate SRs as share of plays with EPA>0; compute quickly from recent window for responsiveness
        return pass_epa, rush_epa, pass_sr, rush_sr

    # compute SRs from recent window by team & type
    sr_map = {teams["home_abbr"]:{True:{"pos":0,"tot":0}, False:{"pos":0,"tot":0}},
              teams["away_abbr"]:{True:{"pos":0,"tot":0}, False:{"pos":0,"tot":0}}}
    for rp in recent:
        # rough heuristic: use description to infer offense team; not perfect, but fine for summary
        # (for SR we don't need exact team here; advanced: re-walk plays with offense team stored)
        pass
    # We'll skip SR from server; the UI focuses on EPA means.

    h_pass_epa = (agg[teams["home_abbr"]]["pass_epa_sum"] / agg[teams["home_abbr"]]["pass_plays"]) if agg[teams["home_abbr"]]["pass_plays"]>0 else 0.0
    h_rush_epa = (agg[teams["home_abbr"]]["rush_epa_sum"] / agg[teams["home_abbr"]]["rush_plays"]) if agg[teams["home_abbr"]]["rush_plays"]>0 else 0.0
    a_pass_epa = (agg[teams["away_abbr"]]["pass_epa_sum"] / agg[teams["away_abbr"]]["pass_plays"]) if agg[teams["away_abbr"]]["pass_plays"]>0 else 0.0
    a_rush_epa = (agg[teams["away_abbr"]]["rush_epa_sum"] / agg[teams["away_abbr"]]["rush_plays"]) if agg[teams["away_abbr"]]["rush_plays"]>0 else 0.0

    # exposure from the newest play clock
    try:
        newest = plays[-1]
        clk = _parse_clock(newest.get("start", {}).get("clock", {}).get("displayValue", "0:00"))
        period = int(newest.get("period", {}).get("number", 0) or 0)
        gsr_cut = _half_secs(period, clk)
    except Exception:
        gsr_cut = 900.0  # fallback mid-game exposure

    # Build feature dict (names must match GLM features)
    feat = {
        "h_off_pass_epa": h_pass_epa, "h_off_rush_epa": h_rush_epa,
        "a_off_pass_epa": a_pass_epa, "a_off_rush_epa": a_rush_epa,

        # Using offense means as defense-allowed proxies
        "a_def_pass_epa": a_pass_epa, "a_def_rush_epa": a_rush_epa,
        "h_def_pass_epa": h_pass_epa, "h_def_rush_epa": h_rush_epa,

        # If your GLM expects SRs, consider adding a rolling SR calc here; we zero-fill for simplicity
        "h_off_pass_sr": 0.0, "h_off_rush_sr": 0.0, "a_off_pass_sr": 0.0, "a_off_rush_sr": 0.0,
        "a_def_pass_sr": 0.0, "a_def_rush_sr": 0.0, "h_def_pass_sr": 0.0, "h_def_rush_sr": 0.0,

        "score_diff_home_cut": float(home_pts - away_pts),
        "score_diff_away_cut": float(away_pts - home_pts),
        "gsr_cut": float(gsr_cut),

        "home_pts_cut": float(home_pts),
        "away_pts_cut": float(away_pts),
    }

    # Predict remaining via GLM
    xh = np.array([feat.get(n, 0.0) for n in NAMES_H], float)
    xa = np.array([feat.get(n, 0.0) for n in NAMES_A], float)
    xh_z = _zscore_with(MU_H, SD_H, xh); xa_z = _zscore_with(MU_A, SD_A, xa)
    rem_h = _predict_remaining(BETA_H, xh_z, exposure=max(1.0, feat["gsr_cut"]))
    rem_a = _predict_remaining(BETA_A, xa_z, exposure=max(1.0, feat["gsr_cut"]))
    pred_home = float(feat["home_pts_cut"] + rem_h)
    pred_away = float(feat["away_pts_cut"] + rem_a)

    # Prepare response
    resp = {
        "teams": {"home_abbr": teams["home_abbr"], "away_abbr": teams["away_abbr"]},
        "score": {"home": int(home_pts), "away": int(away_pts)},
        "predictions": {"home": pred_home, "away": pred_away, "total": pred_home + pred_away},
        "epa": {
            "home": {"off_pass_epa": h_pass_epa, "off_rush_epa": h_rush_epa, "off_epa": (h_pass_epa + h_rush_epa)/2.0},
            "away": {"off_pass_epa": a_pass_epa, "off_rush_epa": a_rush_epa, "off_epa": (a_pass_epa + a_rush_epa)/2.0},
        },
        "recent_plays": list(reversed(recent)),  # newest first in UI
        "updated_at": dt.datetime.utcnow().isoformat() + "Z",
    }
    return resp
