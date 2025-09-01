from pathlib import Path
import json, numpy as np, pandas as pd, joblib

_BASE = Path(__file__).parent
def _load(name): 
    p = _BASE / name
    return joblib.load(str(p)) if p.exists() else None

_M_BLOCK = _load('punt_model_blocked.pkl')
_M_KD    = _load('punt_model_kickdist.pkl')
_M_OUT   = _load('punt_model_outcome.pkl')
_M_RY    = _load('punt_model_returnyards.pkl')

_META = json.load(open(_BASE/'punt_models_meta.json','r'))
_OUT_LABELS = _META['labels_outcome']
_HAVE_MULTI = bool(_META['have_multi_outcome'])
_BASE_FEATS = _META['features_base']
_KD_RMSE = float(_META['metrics'].get('kickdist_rmse', 9.0))
_RY_RMSE = float(_META['metrics'].get('returnyards_rmse', 8.0))

_RNG = np.random.default_rng()

def _row(team, defteam, ydstogo, yard_line, quarter, u2m, score_diff, to_off, to_def):
    return {
        'posteam': str(team), 'defteam': str(defteam), 'ydstogo': float(ydstogo),
        'yardline_100': float(yard_line), 'qtr': int(quarter), 'under_2_minutes': int(u2m),
        'score_differential': float(score_diff), 'posteam_timeouts_remaining': int(to_off),
        'defteam_timeouts_remaining': int(to_def)
    }

def _sample_bool(p, rng): 
    return bool(rng.random() < float(p))


def simulate_punt(team, defteam, ydstogo, yard_line, quarter, under_2_minutes,
                  score_differential, timeouts_off, timeouts_def,
                  decision='sample', random_state=None,
                  add_noise=True, kd_noise=None, ry_noise=None):
    rng = random_state if isinstance(random_state, np.random.Generator) else (np.random.default_rng(random_state) if random_state is not None else _RNG)
    row = _row(team, defteam, ydstogo, yard_line, quarter, under_2_minutes, score_differential, timeouts_off, timeouts_def)
    X = pd.DataFrame([row], columns=_BASE_FEATS)

    p_block = float(_M_BLOCK.predict_proba(X)[0][1]) if _M_BLOCK else 0.01
    blocked = (rng.random() < p_block) if decision=='sample' else (p_block>=0.5)
    if blocked:
        return {'play_type':'punt','blocked':True,'kick_distance':0.0,'outcome':None,'returned':False,'return_yards':0.0}

    kd = float(_M_KD.predict(X)[0]) if _M_KD else 45.0
    sigma_kd = float(kd_noise) if kd_noise is not None else _KD_RMSE
    if add_noise and sigma_kd > 0:
        kd += sigma_kd * rng.standard_normal()
    kd = float(np.clip(kd, 0.0, 80.0))

    if _M_OUT:
        proba = _M_OUT.predict_proba(X)[0]; cls = list(_M_OUT.classes_)
        probs = {c: float(p) for c,p in zip(cls, proba)}
        for k in _OUT_LABELS: probs.setdefault(k, 0.0)
    else:
        probs = {'return':0.4,'fair_catch':0.2,'touchback':0.1,'out_of_bounds':0.15,'downed':0.15} if _HAVE_MULTI else {'return':0.5,'no_return':0.5}

    if _HAVE_MULTI:
        keys = _OUT_LABELS
        p_vec = np.array([probs.get(k,0.0) for k in keys], float); p_vec = p_vec/p_vec.sum() if p_vec.sum()>0 else np.ones(len(keys))/len(keys)
        idx = rng.choice(len(keys), p=p_vec) if decision=='sample' else int(np.argmax(p_vec))
        outcome = keys[idx]; returned = (outcome == 'return')
    else:
        p_ret = float(probs.get('return',0.5))
        returned = (rng.random() < p_ret) if decision=='sample' else (p_ret>=0.5)
        outcome = 'return' if returned else 'no_return'

    ry = 0.0
    if returned:
        ry = float(_M_RY.predict(X)[0]) if _M_RY else 6.0
        sigma_ry = float(ry_noise) if ry_noise is not None else _RY_RMSE
        if add_noise and sigma_ry > 0:
            ry += sigma_ry * rng.standard_normal()
        ry = float(max(0.0, ry))

    return {'play_type':'punt','blocked':False,'kick_distance':kd,'outcome':outcome,'returned':bool(returned),'return_yards':ry}