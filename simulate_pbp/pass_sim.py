
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

_BASE = Path(__file__).parent

def _load(name):
    p = _BASE / name
    return joblib.load(str(p)) if p.exists() else None

_M_SACK = _load('pass_model_sack.pkl')
_M_AIR  = _load('pass_model_air.pkl')
_M_RES  = _load('pass_model_result.pkl')
_M_YAC  = _load('pass_model_yac.pkl')
_M_RF   = _load('pass_model_rec_fumble.pkl')
_M_RFL  = _load('pass_model_rec_fumble_lost.pkl')

_META = json.load(open(_BASE/'pass_models_meta.json','r'))
_BASE_FEATS = _META['features_base']
_LABELS_RES = _META['labels_result']
_AIR_RMSE = float(_META['metrics'].get('air_rmse', 7.0)) if _META['metrics'].get('air_rmse') is not None else 7.0
_YAC_RMSE = float(_META['metrics'].get('yac_rmse', 6.0)) if _META['metrics'].get('yac_rmse') is not None else 6.0

_RNG = np.random.default_rng()

def _row(team, defteam, down, ydstogo, yard_line, quarter, u2m, score_diff, to_off, to_def, pass_location=None, pass_length=None):
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
    if 'pass_location' in _META['categorical']:
        r['pass_location'] = str(pass_location) if pass_location is not None else 'middle'
    if 'pass_length' in _META['categorical']:
        r['pass_length'] = str(pass_length) if pass_length is not None else 'short'
    return r

def simulate_pass(team, defteam, down, ydstogo, yard_line, quarter, under_2_minutes,
                  score_differential, timeouts_off, timeouts_def,
                  pass_location=None, pass_length=None,
                  decision='sample', random_state=None,
                  add_noise=True, air_noise=None, yac_noise=None):
    rng = random_state if isinstance(random_state, np.random.Generator) else (np.random.default_rng(random_state) if random_state is not None else _RNG)
    row = _row(team, defteam, down, ydstogo, yard_line, quarter, under_2_minutes, score_differential, timeouts_off, timeouts_def, pass_location, pass_length)
    cols = list(_BASE_FEATS) + [c for c in ['pass_location','pass_length'] if c in _META['categorical']]
    X = pd.DataFrame([row], columns=cols)

    if _M_SACK is None:
        p_sack = 0.06
    else:
        proba = _M_SACK.predict_proba(X)[0]
        cls = list(_M_SACK.classes_)
        p_sack = float(proba[cls.index(1)]) if 1 in cls else float(proba[-1])
    sack = bool(rng.random() < p_sack) if decision=='sample' else (p_sack >= 0.5)
    if sack:
        sack_yards = -abs(float(_M_YAC.predict(X)[0])) if _M_YAC is not None else -7.0
        return {'play_type':'pass','sack':True,'sack_yards':sack_yards,'air_yards':0.0,'caught':False,'intercepted':False,'dropped':False,'yac':0.0,'yards_gained':sack_yards,'td':False,'receiver_fumble':False,'receiver_fumble_lost':False,'recovery_team':None}

    if _M_AIR is None:
        air = 7.0
    else:
        air = float(_M_AIR.predict(X)[0])
    if add_noise:
        sigma_a = float(air_noise) if air_noise is not None else _AIR_RMSE
        if sigma_a > 0:
            air += sigma_a * rng.standard_normal()
    air = float(np.clip(air, -10.0, yard_line))

    if _M_RES is None:
        probs = {'complete':0.6,'interception':0.03,'incomplete':0.37}
    else:
        proba = _M_RES.predict_proba(X)[0]
        cls = list(_M_RES.classes_)
        probs = {c: float(p) for c,p in zip(cls, proba)}
        for k in _LABELS_RES:
            probs.setdefault(k, 0.0)
    keys = _LABELS_RES
    p_vec = np.array([probs.get(k, 0.0) for k in keys], dtype=float)
    p_vec = p_vec / p_vec.sum() if p_vec.sum()>0 else np.ones(len(keys))/len(keys)
    idx = rng.choice(len(keys), p=p_vec) if decision=='sample' else int(np.argmax(p_vec))
    res = keys[idx]
    caught = (res == 'complete')
    intercepted = (res == 'interception')
    dropped = (res == 'drop')

    if not caught:
        return {'play_type':'pass','sack':False,'sack_yards':0.0,'air_yards':air,'caught':False,'intercepted':intercepted,'dropped':dropped,'yac':0.0,'yards_gained':0.0,'td':False,'receiver_fumble':False,'receiver_fumble_lost':False,'recovery_team':('DEF' if intercepted else None)}

    if _M_YAC is None:
        yac = 4.0
    else:
        yac = float(_M_YAC.predict(X)[0])
    if add_noise:
        sigma_y = float(yac_noise) if yac_noise is not None else _YAC_RMSE
        if sigma_y > 0:
            yac += sigma_y * rng.standard_normal()
    yac = float(max(0.0, yac))

    total = air + yac
    if total >= yard_line:
        yac = max(0.0, yard_line - air)
        yards_gained = float(yard_line)
        td = True
    else:
        yards_gained = float(total)
        td = False

    if _M_RF is None:
        p_f = 0.01
    else:
        proba = _M_RF.predict_proba(X)[0]
        cls = list(_M_RF.classes_)
        p_f = float(proba[cls.index(1)]) if 1 in cls else float(proba[-1])
    receiver_fumble = bool(rng.random() < p_f) if decision=='sample' else (p_f >= 0.5)

    receiver_fumble_lost = False
    recovery_team = team
    if receiver_fumble:
        if _M_RFL is None:
            p_l = 0.5
        else:
            proba = _M_RFL.predict_proba(X)[0]
            cls = list(_M_RFL.classes_)
            p_l = float(proba[cls.index(1)]) if 1 in cls else float(proba[-1])
        receiver_fumble_lost = bool(rng.random() < p_l) if decision=='sample' else (p_l >= 0.5)
        recovery_team = defteam if receiver_fumble_lost else team

    return {'play_type':'pass','sack':False,'sack_yards':0.0,'air_yards':air,'caught':True,'intercepted':False,'dropped':False,'yac':yac,'yards_gained':yards_gained,'td':td,'receiver_fumble':receiver_fumble,'receiver_fumble_lost':receiver_fumble_lost,'recovery_team':recovery_team}
