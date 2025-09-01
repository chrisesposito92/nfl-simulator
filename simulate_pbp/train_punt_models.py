import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import joblib

def _col(df, cands):
    for c in cands:
        if c in df.columns:
            return c
    return None

def _parse_clock(x):
    try:
        m,s = str(x).split(':')
        return int(m)*60 + int(s)
    except Exception:
        return np.nan

def load_engineer(csv_path: str):
    df = pd.read_csv(csv_path, low_memory=False)
    cmap = {}
    cmap['play_type'] = _col(df, ['play_type','PlayType','playType'])
    if cmap['play_type'] is None:
        raise RuntimeError('play_type column missing')
    df = df[df[cmap['play_type']]=='punt'].copy()
    cmap['posteam'] = _col(df, ['posteam','offense_team','offense','OffenseTeam'])
    cmap['defteam'] = _col(df, ['defteam','defense_team','defense','DefenseTeam'])
    cmap['down'] = _col(df, ['down','Down'])
    cmap['ydstogo'] = _col(df, ['ydstogo','YardsToGo','yds_to_go'])
    cmap['yardline_100'] = _col(df, ['yardline_100','YardLine_100','yard_line_100','yardline100'])
    cmap['qtr'] = _col(df, ['qtr','quarter','Quarter'])
    cmap['quarter_seconds_remaining'] = _col(df, ['quarter_seconds_remaining','QuarterSecondsRemaining'])
    cmap['clock'] = _col(df, ['clock','Clock'])
    cmap['score_differential'] = _col(df, ['score_differential','ScoreDiff','scoreDiff'])
    cmap['posteam_timeouts_remaining'] = _col(df, ['posteam_timeouts_remaining','posteam_timeouts','OffenseTimeouts','offense_timeouts_remaining'])
    cmap['defteam_timeouts_remaining'] = _col(df, ['defteam_timeouts_remaining','defteam_timeouts','DefenseTimeouts','defense_timeouts_remaining'])
    cmap['punt_blocked'] = _col(df, ['punt_blocked','blocked_punt','blocked'])
    cmap['kick_distance'] = _col(df, ['kick_distance','punt_distance','punt_yards','punt_dist'])
    cmap['return_yards'] = _col(df, ['return_yards','punt_return_yards','ReturnYards'])
    cmap['fair_catch'] = _col(df, ['fair_catch','faircatch','fair_catch_yards'])
    cmap['touchback'] = _col(df, ['touchback','is_touchback'])
    cmap['out_of_bounds'] = _col(df, ['out_of_bounds','outofbounds','oob'])
    cmap['downed'] = _col(df, ['downed','punt_downed'])
    needed = ['posteam','defteam','ydstogo','yardline_100','qtr']
    miss = [k for k in needed if cmap.get(k) is None]
    if miss:
        raise RuntimeError(f"Missing columns for features: {miss}")
    ren = {}
    for k in ['posteam','defteam','down','ydstogo','yardline_100','qtr']:
        if cmap[k] != k: ren[cmap[k]] = k
    if ren: df = df.rename(columns=ren)
    if cmap['quarter_seconds_remaining'] is None:
        df['quarter_seconds_remaining'] = df[cmap['clock']].map(_parse_clock) if cmap['clock'] else np.nan
    elif cmap['quarter_seconds_remaining'] != 'quarter_seconds_remaining':
        df = df.rename(columns={cmap['quarter_seconds_remaining']:'quarter_seconds_remaining'})
    df['under_2_minutes'] = np.where((df['quarter_seconds_remaining']<=120) & (df[cmap['qtr']].isin([2,4])), 1, 0)
    if cmap['score_differential'] is None:
        df['score_differential'] = 0.0
    elif cmap['score_differential'] != 'score_differential':
        df = df.rename(columns={cmap['score_differential']:'score_differential'})
    if cmap['posteam_timeouts_remaining'] is None:
        df['posteam_timeouts_remaining'] = 3
    elif cmap['posteam_timeouts_remaining'] != 'posteam_timeouts_remaining':
        df = df.rename(columns={cmap['posteam_timeouts_remaining']:'posteam_timeouts_remaining'})
    if cmap['defteam_timeouts_remaining'] is None:
        df['defteam_timeouts_remaining'] = 3
    elif cmap['defteam_timeouts_remaining'] != 'defteam_timeouts_remaining':
        df = df.rename(columns={cmap['defteam_timeouts_remaining']:'defteam_timeouts_remaining'})
    for k in ['punt_blocked','fair_catch','touchback','out_of_bounds','downed']:
        if cmap[k] is None:
            df[k] = 0
        else:
            if cmap[k] != k:
                df = df.rename(columns={cmap[k]:k})
            df[k] = df[k].fillna(0).astype(int)
    if cmap['kick_distance'] is None:
        df['kick_distance'] = np.nan
    elif cmap['kick_distance'] != 'kick_distance':
        df = df.rename(columns={cmap['kick_distance']:'kick_distance'})
    if cmap['return_yards'] is None:
        df['return_yards'] = np.nan
    elif cmap['return_yards'] != 'return_yards':
        df = df.rename(columns={cmap['return_yards']:'return_yards'})
    df = df.dropna(subset=['posteam','defteam','ydstogo','yardline_100','qtr'])
    df['ydstogo'] = df['ydstogo'].astype(float)
    df['yardline_100'] = df['yardline_100'].astype(float)
    df['qtr'] = df['qtr'].astype(int)
    df['under_2_minutes'] = df['under_2_minutes'].astype(int)
    df['score_differential'] = df['score_differential'].astype(float)
    df['posteam_timeouts_remaining'] = df['posteam_timeouts_remaining'].astype(int)
    df['defteam_timeouts_remaining'] = df['defteam_timeouts_remaining'].astype(int)
    have_multi = all(c in df.columns for c in ['fair_catch','touchback','out_of_bounds','downed'])
    if have_multi:
        outcome = np.where(df['fair_catch']==1, 'fair_catch',
                   np.where(df['touchback']==1, 'touchback',
                   np.where(df['out_of_bounds']==1, 'out_of_bounds',
                   np.where(df['downed']==1, 'downed', 'return'))))
        df['punt_outcome'] = outcome
    else:
        df['punt_outcome'] = np.where(df['return_yards'].fillna(-1) >= 0, 'return', 'no_return')
    df['returned_flag'] = (df['punt_outcome']=='return').astype(int)
    return df, have_multi

def _clf(cat_cols, num_cols):
    pre = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
                             ('num', 'passthrough', num_cols)])
    clf = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1, class_weight='balanced')
    return Pipeline([('pre', pre), ('clf', clf)])

def _reg(cat_cols, num_cols, kind='rf'):
    pre = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
                             ('num', 'passthrough', num_cols)])
    if kind=='rf':
        reg = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    else:
        reg = GradientBoostingRegressor(random_state=42)
    return Pipeline([('pre', pre), ('reg', reg)])

def _safe_split(X, y, test_size, random_state):
    if test_size <= 0 or len(X) == 0:
        return X, X.iloc[0:0], y, y.iloc[0:0]
    vc = pd.Series(y).value_counts()
    if (vc.min() if len(vc)>0 else 0) < 2 or y.nunique() < 2:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True, stratify=None)
    return train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True, stratify=y)

def train_models(df, have_multi, outdir: Path, val_size: float, random_state: int):
    outdir.mkdir(parents=True, exist_ok=True)
    base = ['posteam','defteam','ydstogo','yardline_100','qtr','under_2_minutes','score_differential','posteam_timeouts_remaining','defteam_timeouts_remaining']
    cat = ['posteam','defteam','qtr']
    num = [c for c in base if c not in cat]
    y_block = df['punt_blocked'].astype(int)
    X = df[base]
    trX, teX, trY, teY = _safe_split(X, y_block, val_size, random_state)
    m_block = _clf(cat, num)
    m_block.fit(trX, trY)
    yhat = m_block.predict(teX) if len(teX) else np.array([])
    acc_block = float(accuracy_score(teY, yhat)) if len(teX) else None
    joblib.dump(m_block, outdir/'punt_model_blocked.pkl')
    has_kd = df['kick_distance'].notna().sum() > 50
    if has_kd:
        y_kd = df['kick_distance'].astype(float)
        trX, teX, trY, teY = _safe_split(df[base], y_kd, val_size, random_state)
        m_kd = _reg(cat, num, kind='gbr')
        m_kd.fit(trX, trY)
        ypred = m_kd.predict(teX) if len(teX) else np.array([])
        mae_kd = float(mean_absolute_error(teY, ypred)) if len(teX) else None
        rmse_kd = float(np.sqrt(mean_squared_error(teY, ypred))) if len(teX) else None
        joblib.dump(m_kd, outdir/'punt_model_kickdist.pkl')
    else:
        mae_kd = None; rmse_kd = None
    if have_multi:
        y_out = df['punt_outcome']
        trX, teX, trY, teY = _safe_split(df[base], y_out, val_size, random_state)
        m_out = _clf(cat, num)
    else:
        y_out = (df['punt_outcome']=='return').astype(int)
        trX, teX, trY, teY = _safe_split(df[base], y_out, val_size, random_state)
        m_out = _clf(cat, num)
    m_out.fit(trX, trY)
    yhat = m_out.predict(teX) if len(teX) else np.array([])
    acc_out = float(accuracy_score(teY, yhat)) if len(teX) else None
    joblib.dump(m_out, outdir/'punt_model_outcome.pkl')
    df_ret = df[(df['returned_flag']==1) & df['return_yards'].notna()].copy()
    if len(df_ret) >= 50:
        y_ry = df_ret['return_yards'].astype(float).clip(lower=0.0)
        trX, teX, trY, teY = _safe_split(df_ret[base], y_ry, val_size, random_state)
        m_ry = _reg(cat, num, kind='rf')
        m_ry.fit(trX, trY)
        ypred = m_ry.predict(teX) if len(teX) else np.array([])
        mae_ry = float(mean_absolute_error(teY, ypred)) if len(teX) else None
        rmse_ry = float(np.sqrt(mean_squared_error(teY, ypred))) if len(teX) else None
        joblib.dump(m_ry, outdir/'punt_model_returnyards.pkl')
    else:
        mae_ry = None; rmse_ry = None
    meta = {
        "features_base": base,
        "categorical": cat,
        "have_multi_outcome": bool(have_multi),
        "labels_outcome": (sorted(df['punt_outcome'].unique().tolist()) if have_multi else ["no_return","return"]),
        "metrics": {
            "blocked_acc": acc_block,
            "kickdist_mae": mae_kd,
            "kickdist_rmse": rmse_kd,
            "outcome_acc": acc_out,
            "returnyards_mae": mae_ry,
            "returnyards_rmse": rmse_ry
        }
    }
    with open(outdir/'punt_models_meta.json','w') as f:
        json.dump(meta, f, indent=2)
    return meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=str, required=True)
    ap.add_argument('--outdir', type=str, default='.')
    ap.add_argument('--val_size', type=float, default=0.2)
    ap.add_argument('--random_state', type=int, default=42)
    args = ap.parse_args()
    df, have_multi = load_engineer(args.csv)
    outdir = Path(args.outdir)
    meta = train_models(df, have_multi, outdir, args.val_size, args.random_state)
    print(json.dumps(meta["metrics"], indent=2))

if __name__ == '__main__':
    main()