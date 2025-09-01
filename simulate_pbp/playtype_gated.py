from pathlib import Path
import joblib
import pandas as pd
import numpy as np

_BASE = Path(__file__).parent

def _load(p):
    p = _BASE / p
    return joblib.load(str(p)) if p.exists() else None

_M_EARLY = _load('model_early_pass_run.pkl')
_M_4_KICK3 = _load('model_4th_kick3.pkl')
_M_4_PG = _load('model_4th_punt_go.pkl')
_M_4_PR = _load('model_4th_pass_run.pkl')

def _fg_distance(yard_line, snap=17.0):
    return float(yard_line) + float(snap)

def _decide_label(probs, decision='argmax', temperature=1.0, epsilon=0.0, rng=None):
    keys = ['field_goal','punt','pass','run']
    p = np.array([probs.get(k,0.0) for k in keys], dtype=float)
    s = float(p.sum())
    if s <= 0:
        p = np.array([0.0,0.0,0.5,0.5], dtype=float)
    else:
        p /= s
    if decision == 'sample':
        T = float(temperature) if temperature and temperature > 0 else 1.0
        if T != 1.0:
            p = np.power(p, 1.0/T)
            p /= p.sum()
        rng = np.random.default_rng(rng)
        idx = rng.choice(len(keys), p=p)
        return keys[idx]
    if decision == 'epsilon_greedy':
        rng = np.random.default_rng(rng)
        if rng.random() < float(epsilon):
            idx = rng.choice(len(keys), p=p)
            return keys[idx]
    return keys[int(np.argmax(p))]

def predict_play_type(team, down, yards_to_go, yard_line, quarter, under_2_minutes,
                      score_differential, timeouts_off, timeouts_def,
                      fg_max_distance=70.0, fg_snap_distance=17.0,
                      decision='argmax', temperature=1.0, epsilon=0.0, random_state=None):
    row = {
        'posteam': str(team),
        'down': int(down),
        'ydstogo': float(yards_to_go),
        'yardline_100': float(yard_line),
        'qtr': int(quarter),
        'under_2_minutes': int(under_2_minutes),
        'score_differential': float(score_differential),
        'posteam_timeouts_remaining': int(timeouts_off),
        'defteam_timeouts_remaining': int(timeouts_def),
    }

    if row['down'] in (1,2,3):
        X = pd.DataFrame([row], columns=[
            'posteam','down','ydstogo','yardline_100','qtr',
            'under_2_minutes','score_differential',
            'posteam_timeouts_remaining','defteam_timeouts_remaining'
        ])
        proba = _M_EARLY.predict_proba(X)[0] if _M_EARLY is not None else np.array([0.5,0.5])
        classes = list(_M_EARLY.classes_) if _M_EARLY is not None else ['pass','run']
        d = {c: float(p) for c,p in zip(classes, proba)}
        probs = {'pass': d.get('pass',0.0), 'run': d.get('run',0.0), 'punt': 0.0, 'field_goal': 0.0}
        label = _decide_label(probs, decision, temperature, epsilon, random_state)
        return label, probs, {'gate':'early'}

    fg_dist = _fg_distance(row['yardline_100'], fg_snap_distance)
    feats = [
        'posteam','down','ydstogo','yardline_100','qtr',
        'under_2_minutes','score_differential',
        'posteam_timeouts_remaining','defteam_timeouts_remaining','fg_distance'
    ]
    row2 = dict(row); row2['fg_distance'] = fg_dist

    if fg_dist <= fg_max_distance and _M_4_KICK3 is not None:
        X = pd.DataFrame([row2], columns=feats)
        proba = _M_4_KICK3.predict_proba(X)[0]
        classes = list(_M_4_KICK3.classes_)
        d = {c: float(p) for c,p in zip(classes, proba)}
        p_fg = d.get('field_goal',0.0)
        p_punt = d.get('punt',0.0)
        p_go = d.get('go',0.0)

        if _M_4_PR is not None and p_go > 0.0:
            proba_pr = _M_4_PR.predict_proba(X)[0]
            classes_pr = list(_M_4_PR.classes_)
            dpr = {c: float(p) for c,p in zip(classes_pr, proba_pr)}
            p_pass = p_go * dpr.get('pass',0.0)
            p_run  = p_go * dpr.get('run',0.0)
        else:
            p_pass = p_run = p_go / 2.0

        total = p_fg + p_punt + p_pass + p_run
        total = total if total > 0 else 1.0
        probs = {
            'field_goal': p_fg/total,
            'punt': p_punt/total,
            'pass': p_pass/total,
            'run': p_run/total
        }
        label = _decide_label(probs, decision, temperature, epsilon, random_state)
        return label, probs, {'gate':'4th_fg_window','fg_distance':fg_dist}

    if _M_4_PG is None:
        p_punt, p_go = 0.5, 0.5
    else:
        X = pd.DataFrame([row2], columns=feats)
        proba = _M_4_PG.predict_proba(X)[0]
        classes = list(_M_4_PG.classes_)
        d = {c: float(p) for c,p in zip(classes, proba)}
        p_punt = d.get('punt',0.0)
        p_go   = d.get('go',0.0)

    if _M_4_PR is not None and p_go > 0.0:
        proba_pr = _M_4_PR.predict_proba(X)[0]
        classes_pr = list(_M_4_PR.classes_)
        dpr = {c: float(p) for c,p in zip(classes_pr, proba_pr)}
        p_pass = p_go * dpr.get('pass',0.0)
        p_run  = p_go * dpr.get('run',0.0)
    else:
        p_pass = p_run = p_go / 2.0

    total = p_punt + p_pass + p_run
    total = total if total > 0 else 1.0
    probs = {
        'field_goal': 0.0,
        'punt': p_punt/total,
        'pass': p_pass/total,
        'run': p_run/total
    }
    label = _decide_label(probs, decision, temperature, epsilon, random_state)
    return label, probs, {'gate':'4th_no_fg','fg_distance':fg_dist}