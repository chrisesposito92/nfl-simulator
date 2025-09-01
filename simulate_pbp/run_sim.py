from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

_BASE = Path(__file__).parent

def _load(name):
    p = _BASE / name
    return joblib.load(str(p)) if p.exists() else None

_M_YARDS = _load('run_model_yards.pkl')
_M_FUM   = _load('run_model_fumble.pkl')
_M_FL    = _load('run_model_fumble_lost.pkl')
_M_OOB   = _load('run_model_oob.pkl')

_META = json.load(open(_BASE/'run_models_meta.json','r'))
_BASE_FEATS = _META['features_base']
_HAVE_OOB = bool(_META['have_oob'])
_YARDS_RMSE = float(_META['metrics'].get('yards_rmse', 7.0))

_RNG = np.random.default_rng()

def _row(team, defteam, down, ydstogo, yard_line, quarter, u2m, score_diff, to_off, to_def, run_location=None, run_gap=None):
    r = {
        'posteam': str(team),
        'defteam': str(defteam),
        'down': int(down),
        'ydstogo': float(ydstogo),
        'yardline_100': float(yard_line),
        'qtr': int(quarter),
        'under_2_minutes': int(u2m),
        'score_differential': float(score_diff),
        'posteam_timeouts_remaining': int(to_off),
        'defteam_timeouts_remaining': int(to_def)
    }
    if 'run_location' in _META['categorical']:
        r['run_location'] = str(run_location) if run_location is not None else 'middle'
    if 'run_gap' in _META['categorical']:
        r['run_gap'] = str(run_gap) if run_gap is not None else 'guard'
    return r

def simulate_run(team, defteam, down, ydstogo, yard_line, quarter, under_2_minutes,
                 score_differential, timeouts_off, timeouts_def,
                 run_location=None, run_gap=None,
                 decision='sample', random_state=None,
                 add_noise=True, yards_noise=None,
                 min_yards=-10.0, max_yards=99.0):
    rng = random_state if isinstance(random_state, np.random.Generator) else (np.random.default_rng(random_state) if random_state is not None else _RNG)
    row = _row(team, defteam, down, ydstogo, yard_line, quarter, under_2_minutes, score_differential, timeouts_off, timeouts_def, run_location, run_gap)
    cols = list(_BASE_FEATS) + [c for c in ['run_location','run_gap'] if c in _META['categorical']]
    X = pd.DataFrame([row], columns=cols)
    y = float(_M_YARDS.predict(X)[0]) if _M_YARDS is not None else 4.0
    sigma = float(yards_noise) if yards_noise is not None else _YARDS_RMSE
    if add_noise and sigma > 0:
        y += sigma * rng.standard_normal()
    y = float(np.clip(y, min_yards, max_yards))
    td = bool(y >= float(yard_line))
    if td:
        y = float(yard_line)
    if _M_FUM is None:
        p_fum = 0.015
    else:
        proba = _M_FUM.predict_proba(X)[0]
        cls = list(_M_FUM.classes_)
        p_fum = float(proba[cls.index(1)]) if 1 in cls else float(proba[-1])
    fumble = bool(rng.random() < p_fum) if decision=='sample' else (p_fum >= 0.5)
    fumble_lost = False
    recovery_team = team
    if fumble:
        if _M_FL is None:
            p_lost = 0.5
        else:
            proba = _M_FL.predict_proba(X)[0]
            cls = list(_M_FL.classes_)
            p_lost = float(proba[cls.index(1)]) if 1 in cls else float(proba[-1])
        fumble_lost = bool(rng.random() < p_lost) if decision=='sample' else (p_lost >= 0.5)
        recovery_team = defteam if fumble_lost else team
    if _HAVE_OOB and _M_OOB is not None:
        proba = _M_OOB.predict_proba(X)[0]
        cls = list(_M_OOB.classes_)
        p_oob = float(proba[cls.index(1)]) if 1 in cls else float(proba[-1])
        oob = bool(rng.random() < p_oob) if decision=='sample' else (p_oob >= 0.5)
    else:
        oob = bool(rng.random() < 0.2) if decision=='sample' else False
    return {
        'play_type': 'run',
        'yards_gained': y,
        'td': td,
        'fumble': fumble,
        'fumble_lost': fumble_lost,
        'recovery_team': recovery_team,
        'out_of_bounds': oob
    }