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
        m, s = str(x).split(':')
        return int(m)*60 + int(s)
    except Exception:
        return np.nan

def load_engineer(csv_path: str):
    df = pd.read_csv(csv_path, low_memory=False)

    cmap = {}
    cmap['play_type'] = _col(df, ['play_type','PlayType','playType'])
    if cmap['play_type'] is None:
        raise RuntimeError('play_type column missing')
    df = df[df[cmap['play_type']]=='run'].copy()

    # optional filters for nullified plays
    no_play = _col(df, ['no_play','NoPlay'])
    if no_play:
        df = df[df[no_play]!=1]

    cmap['posteam'] = _col(df, ['posteam','offense_team','offense','OffenseTeam'])
    cmap['defteam'] = _col(df, ['defteam','defense_team','defense','DefenseTeam'])
    cmap['down'] = _col(df, ['down','Down'])
    cmap['ydstogo'] = _col(df, ['ydstogo','YardsToGo','yds_to_go'])
    cmap['yardline_100'] = _col(df, ['yardline_100','YardLine_100','yard_line_100','yardline100'])
    cmap['qtr'] = _col(df, ['qtr','quarter','Quarter'])
    cmap['clock'] = _col(df, ['clock','Clock'])
    cmap['quarter_seconds_remaining'] = _col(df, ['quarter_seconds_remaining','QuarterSecondsRemaining'])
    cmap['score_differential'] = _col(df, ['score_differential','ScoreDiff','scoreDiff'])
    cmap['posteam_timeouts_remaining'] = _col(df, ['posteam_timeouts_remaining','posteam_timeouts','OffenseTimeouts','offense_timeouts_remaining'])
    cmap['defteam_timeouts_remaining'] = _col(df, ['defteam_timeouts_remaining','defteam_timeouts','DefenseTimeouts','defense_timeouts_remaining'])

    # targets
    cmap['yards_gained'] = _col(df, ['yards_gained','rush_yards','yards','rushing_yards'])
    cmap['fumble'] = _col(df, ['fumble','rush_fumble','fumbled'])
    cmap['fumble_lost'] = _col(df, ['fumble_lost','lost_fumble'])
    cmap['out_of_bounds'] = _col(df, ['out_of_bounds','oob'])
    # optional categorical detail
    cmap['run_location'] = _col(df, ['run_location','rush_location'])
    cmap['run_gap'] = _col(df, ['run_gap','rush_gap'])

    need = ['posteam','defteam','down','ydstogo','yardline_100','qtr']
    miss = [k for k in need if cmap.get(k) is None]
    if miss:
        raise RuntimeError(f"Missing columns: {miss}")

    # canonicalize
    ren = {}
    for k in ['posteam','defteam','down','ydstogo','yardline_100','qtr']:
        if cmap[k] != k: ren[cmap[k]] = k
    if ren: df = df.rename(columns=ren)

    # time left
    if cmap['quarter_seconds_remaining'] is None:
        df['quarter_seconds_remaining'] = df[cmap['clock']].map(_parse_clock) if cmap['clock'] else np.nan
    elif cmap['quarter_seconds_remaining'] != 'quarter_seconds_remaining':
        df = df.rename(columns={cmap['quarter_seconds_remaining']: 'quarter_seconds_remaining'})
    df['under_2_minutes'] = np.where((df['quarter_seconds_remaining']<=120) & (df['qtr'].isin([2,4])), 1, 0)

    # score and timeouts
    if cmap['score_differential'] is None:
        df['score_differential'] = 0.0
    elif cmap['score_differential'] != 'score_differential':
        df = df.rename(columns={cmap['score_differential']: 'score_differential'})
    if cmap['posteam_timeouts_remaining'] is None:
        df['posteam_timeouts_remaining'] = 3
    elif cmap['posteam_timeouts_remaining'] != 'posteam_timeouts_remaining':
        df = df.rename(columns={cmap['posteam_timeouts_remaining']: 'posteam_timeouts_remaining'})
    if cmap['defteam_timeouts_remaining'] is None:
        df['defteam_timeouts_remaining'] = 3
    elif cmap['defteam_timeouts_remaining'] != 'defteam_timeouts_remaining':
        df = df.rename(columns={cmap['defteam_timeouts_remaining']: 'defteam_timeouts_remaining'})

    # targets: yards
    if cmap['yards_gained'] is None:
        raise RuntimeError('yards_gained column missing for run plays')
    if cmap['yards_gained'] != 'yards_gained':
        df = df.rename(columns={cmap['yards_gained']: 'yards_gained'})

    # targets: fumble
    if cmap['fumble'] is None and cmap['fumble_lost'] is None:
        df['fumble'] = 0
    else:
        if cmap['fumble'] and cmap['fumble'] != 'fumble':
            df = df.rename(columns={cmap['fumble']: 'fumble'})
        if cmap['fumble_lost'] and cmap['fumble_lost'] != 'fumble_lost':
            df = df.rename(columns={cmap['fumble_lost']: 'fumble_lost'})
        if 'fumble' not in df.columns: df['fumble'] = 0
        if 'fumble_lost' not in df.columns: df['fumble_lost'] = 0
        df['fumble'] = ((df['fumble']==1) | (df['fumble_lost']==1)).astype(int)

    # targets: out_of_bounds
    have_oob = False
    if cmap['out_of_bounds']:
        if cmap['out_of_bounds'] != 'out_of_bounds':
            df = df.rename(columns={cmap['out_of_bounds']:'out_of_bounds'})
        df['out_of_bounds'] = df['out_of_bounds'].fillna(0).astype(int)
        have_oob = True

    # optional categorical features
    if cmap['run_location'] and cmap['run_location'] != 'run_location':
        df = df.rename(columns={cmap['run_location']:'run_location'})
    if cmap['run_gap'] and cmap['run_gap'] != 'run_gap':
        df = df.rename(columns={cmap['run_gap']:'run_gap'})

    # clean types
    df = df.dropna(subset=['posteam','defteam','down','ydstogo','yardline_100','qtr','yards_gained'])
    df['down'] = df['down'].astype(int)
    df['ydstogo'] = df['ydstogo'].astype(float)
    df['yardline_100'] = df['yardline_100'].astype(float)
    df['qtr'] = df['qtr'].astype(int)
    df['under_2_minutes'] = df['under_2_minutes'].astype(int)
    df['score_differential'] = df['score_differential'].astype(float)
    df['posteam_timeouts_remaining'] = df['posteam_timeouts_remaining'].astype(int)
    df['defteam_timeouts_remaining'] = df['defteam_timeouts_remaining'].astype(int)
    df['yards_gained'] = df['yards_gained'].astype(float)

    return df, have_oob

def _clf(cat_cols, num_cols):
    pre = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
                             ('num', 'passthrough', num_cols)])
    clf = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1, class_weight='balanced')
    return Pipeline([('pre', pre), ('clf', clf)])

def _reg(cat_cols, num_cols, kind='gbr'):
    pre = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
                             ('num', 'passthrough', num_cols)])
    if kind == 'gbr':
        reg = GradientBoostingRegressor(random_state=42)
    else:
        reg = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    return Pipeline([('pre', pre), ('reg', reg)])

def _safe_split(X, y, test_size, random_state, stratify_ok=True):
    if test_size <= 0 or len(X) == 0:
        return X, X.iloc[0:0], y, y.iloc[0:0]
    if stratify_ok and hasattr(y, 'nunique') and y.nunique() > 1:
        vc = pd.Series(y).value_counts()
        if (vc.min() if len(vc)>0 else 0) >= 2:
            return train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True, stratify=y)
    return train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True)

def train_all(df, have_oob, outdir: Path, val_size: float, random_state: int):
    outdir.mkdir(parents=True, exist_ok=True)

    base = ['posteam','defteam','down','ydstogo','yardline_100','qtr','under_2_minutes',
            'score_differential','posteam_timeouts_remaining','defteam_timeouts_remaining']
    cat = ['posteam','defteam','qtr']
    if 'run_location' in df.columns: cat.append('run_location')
    if 'run_gap' in df.columns: cat.append('run_gap')
    num = [c for c in base if c not in cat]

    # yards_gained regression
    X = df[base + [c for c in ['run_location','run_gap'] if c in df.columns]]
    y = df['yards_gained']
    trX, teX, trY, teY = _safe_split(X, y, val_size, random_state, stratify_ok=False)
    m_yards = _reg(cat, num, kind='gbr')
    m_yards.fit(trX, trY)
    yhat = m_yards.predict(teX) if len(teX) else np.array([])
    mae_y = float(mean_absolute_error(teY, yhat)) if len(teX) else None
    rmse_y = float(np.sqrt(mean_squared_error(teY, yhat))) if len(teX) else None
    joblib.dump(m_yards, outdir/'run_model_yards.pkl')

    # fumble classification
    y = df['fumble'].astype(int)
    trX, teX, trY, teY = _safe_split(X, y, val_size, random_state, stratify_ok=True)
    m_fum = _clf(cat, num)
    m_fum.fit(trX, trY)
    yhat = m_fum.predict(teX) if len(teX) else np.array([])
    acc_f = float(accuracy_score(teY, yhat)) if len(teX) else None
    joblib.dump(m_fum, outdir/'run_model_fumble.pkl')

    # out_of_bounds classification (optional)
    acc_oob = None
    if have_oob:
        y = df['out_of_bounds'].astype(int)
        trX, teX, trY, teY = _safe_split(X, y, val_size, random_state, stratify_ok=True)
        m_oob = _clf(cat, num)
        m_oob.fit(trX, trY)
        yhat = m_oob.predict(teX) if len(teX) else np.array([])
        acc_oob = float(accuracy_score(teY, yhat)) if len(teX) else None
        joblib.dump(m_oob, outdir/'run_model_oob.pkl')

    meta = {
        "features_base": base,
        "categorical": cat,
        "have_oob": bool(have_oob),
        "metrics": {
            "yards_mae": mae_y,
            "yards_rmse": rmse_y,
            "fumble_acc": acc_f,
            "oob_acc": acc_oob
        }
    }
    with open(outdir/'run_models_meta.json','w') as f:
        json.dump(meta, f, indent=2)
    return meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=str, required=True)
    ap.add_argument('--outdir', type=str, default='.')
    ap.add_argument('--val_size', type=float, default=0.2)
    ap.add_argument('--random_state', type=int, default=42)
    args = ap.parse_args()
    df, have_oob = load_engineer(args.csv)
    meta = train_all(df, have_oob, Path(args.outdir), args.val_size, args.random_state)
    print(json.dumps(meta["metrics"], indent=2))

if __name__ == '__main__':
    main()