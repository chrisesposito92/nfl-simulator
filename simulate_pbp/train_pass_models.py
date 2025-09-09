
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
    df = df[df[cmap['play_type']]=='pass'].copy()

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

    cmap['sack'] = _col(df, ['sack','qb_sack','is_sack'])
    cmap['yards_gained'] = _col(df, ['yards_gained','yards','pass_yards'])
    cmap['air_yards'] = _col(df, ['air_yards','AirYards','pass_air_yards'])
    cmap['yards_after_catch'] = _col(df, ['yards_after_catch','yac','YardsAfterCatch'])
    cmap['complete_pass'] = _col(df, ['complete_pass','pass_complete','is_complete'])
    cmap['incomplete_pass'] = _col(df, ['incomplete_pass','pass_incomplete','is_incomplete'])
    cmap['interception'] = _col(df, ['interception','intercepted','is_interception'])
    cmap['receiver_drop'] = _col(df, ['receiver_drop','drop','dropped','pass_dropped'])
    cmap['pass_location'] = _col(df, ['pass_location','pass_loc'])
    cmap['pass_length'] = _col(df, ['pass_length','pass_depth','depth'])

    cmap['fumble'] = _col(df, ['fumble','fumbled'])
    cmap['fumble_lost'] = _col(df, ['fumble_lost','lost_fumble'])
    cmap['pass_touchdown'] = _col(df, ['pass_touchdown','passing_td','touchdown'])

    need = ['posteam','defteam','down','ydstogo','yardline_100','qtr']
    miss = [k for k in need if cmap.get(k) is None]
    if miss:
        raise RuntimeError(f"Missing columns: {miss}")

    ren = {}
    for k in ['posteam','defteam','down','ydstogo','yardline_100','qtr']:
        if cmap[k] != k: ren[cmap[k]] = k
    if ren: df = df.rename(columns=ren)

    if cmap['quarter_seconds_remaining'] is None:
        df['quarter_seconds_remaining'] = df[cmap['clock']].map(_parse_clock) if cmap['clock'] else np.nan
    elif cmap['quarter_seconds_remaining'] != 'quarter_seconds_remaining':
        df = df.rename(columns={cmap['quarter_seconds_remaining']:'quarter_seconds_remaining'})
    df['under_2_minutes'] = np.where((df['quarter_seconds_remaining']<=120) & (df['qtr'].isin([2,4])), 1, 0)

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

    if cmap['sack'] and cmap['sack'] != 'sack':
        df = df.rename(columns={cmap['sack']:'sack'})
    if 'sack' not in df.columns:
        df['sack'] = 0

    if cmap['yards_gained'] and cmap['yards_gained'] != 'yards_gained':
        df = df.rename(columns={cmap['yards_gained']:'yards_gained'})
    if 'yards_gained' not in df.columns:
        df['yards_gained'] = np.nan

    if cmap['air_yards'] and cmap['air_yards'] != 'air_yards':
        df = df.rename(columns={cmap['air_yards']:'air_yards'})
    if 'air_yards' not in df.columns:
        df['air_yards'] = np.nan

    if cmap['yards_after_catch'] and cmap['yards_after_catch'] != 'yards_after_catch':
        df = df.rename(columns={cmap['yards_after_catch']:'yards_after_catch'})

    if cmap['complete_pass'] and cmap['complete_pass'] != 'complete_pass':
        df = df.rename(columns={cmap['complete_pass']:'complete_pass'})
    if 'complete_pass' not in df.columns:
        df['complete_pass'] = np.nan

    if cmap['interception'] and cmap['interception'] != 'interception':
        df = df.rename(columns={cmap['interception']:'interception'})
    if 'interception' not in df.columns:
        df['interception'] = 0

    if cmap['incomplete_pass'] and cmap['incomplete_pass'] != 'incomplete_pass':
        df = df.rename(columns={cmap['incomplete_pass']:'incomplete_pass'})

    if cmap['receiver_drop'] and cmap['receiver_drop'] != 'receiver_drop':
        df = df.rename(columns={cmap['receiver_drop']:'receiver_drop'})

    if cmap['pass_location'] and cmap['pass_location'] != 'pass_location':
        df = df.rename(columns={cmap['pass_location']:'pass_location'})
    if cmap['pass_length'] and cmap['pass_length'] != 'pass_length':
        df = df.rename(columns={cmap['pass_length']:'pass_length'})

    if cmap['fumble'] and cmap['fumble'] != 'fumble':
        df = df.rename(columns={cmap['fumble']:'fumble'})
    if 'fumble' not in df.columns:
        df['fumble'] = 0

    if cmap['fumble_lost'] and cmap['fumble_lost'] != 'fumble_lost':
        df = df.rename(columns={cmap['fumble_lost']:'fumble_lost'})

    if cmap['pass_touchdown'] and cmap['pass_touchdown'] != 'pass_touchdown':
        df = df.rename(columns={cmap['pass_touchdown']:'pass_touchdown'})

    df = df.dropna(subset=['posteam','defteam','down','ydstogo','yardline_100','qtr']).copy()
    df['down'] = df['down'].astype(int)
    df['ydstogo'] = df['ydstogo'].astype(float)
    df['yardline_100'] = df['yardline_100'].astype(float)
    df['qtr'] = df['qtr'].astype(int)
    df['under_2_minutes'] = df['under_2_minutes'].astype(int)
    df['score_differential'] = df['score_differential'].astype(float)
    df['posteam_timeouts_remaining'] = df['posteam_timeouts_remaining'].astype(int)
    df['defteam_timeouts_remaining'] = df['defteam_timeouts_remaining'].astype(int)
    df['sack'] = df['sack'].fillna(0).astype(int)
    if 'complete_pass' in df.columns:
        df['complete_pass'] = df['complete_pass'].fillna(0).astype(int)
    df['interception'] = df['interception'].fillna(0).astype(int)
    if 'incomplete_pass' in df.columns:
        df['incomplete_pass'] = df['incomplete_pass'].fillna(0).astype(int)
    if 'receiver_drop' in df.columns:
        df['receiver_drop'] = df['receiver_drop'].fillna(0).astype(int)
    if 'fumble_lost' in df.columns:
        df['fumble_lost'] = df['fumble_lost'].fillna(0).astype(int)
    if 'pass_location' in df.columns:
        df['pass_location'] = df['pass_location'].fillna('middle').astype(str)
    if 'pass_length' in df.columns:
        df['pass_length'] = df['pass_length'].fillna('short').astype(str)

    if 'yards_after_catch' not in df.columns or df['yards_after_catch'].isna().all():
        yac = df['yards_gained'] - df['air_yards'].fillna(0)
        yac = yac.clip(lower=0)
        df['yards_after_catch'] = yac

    return df

def _clf(cat_cols, num_cols):
    pre = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
                             ('num', 'passthrough', num_cols)])
    clf = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1, class_weight='balanced')
    return Pipeline([('pre', pre), ('clf', clf)])

def _reg(cat_cols, num_cols, kind='gbr'):
    pre = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
                             ('num', 'passthrough', num_cols)])
    reg = GradientBoostingRegressor(random_state=42) if kind=='gbr' else RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    return Pipeline([('pre', pre), ('reg', reg)])

def _safe_split(X, y, test_size, random_state, stratify_ok=True):
    if test_size <= 0 or len(X) == 0:
        return X, X.iloc[0:0], y, y.iloc[0:0]
    if stratify_ok and hasattr(y, 'nunique') and y.nunique() > 1:
        vc = pd.Series(y).value_counts()
        if (vc.min() if len(vc)>0 else 0) >= 2:
            return train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True, stratify=y)
    return train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True)

def train_all(df, outdir: Path, val_size: float, random_state: int):
    outdir.mkdir(parents=True, exist_ok=True)

    base = ['posteam','defteam','down','ydstogo','yardline_100','qtr','under_2_minutes','score_differential','posteam_timeouts_remaining','defteam_timeouts_remaining']
    cat = ['posteam','defteam','qtr']
    if 'pass_location' in df.columns: cat.append('pass_location')
    if 'pass_length' in df.columns: cat.append('pass_length')
    num = [c for c in base if c not in cat]

    X_base = df[base + [c for c in ['pass_location','pass_length'] if c in df.columns]]

    y_sack = df['sack'].astype(int)
    trX, teX, trY, teY = _safe_split(X_base, y_sack, val_size, random_state, stratify_ok=True)
    m_sack = _clf(cat, num)
    m_sack.fit(trX, trY)
    yhat = m_sack.predict(teX) if len(teX) else np.array([])
    acc_sack = float(accuracy_score(teY, yhat)) if len(teX) else None
    joblib.dump(m_sack, outdir/'pass_model_sack.pkl')

    df_nosack = df[df['sack']==0].copy()
    mask_air = df_nosack['air_yards'].notna()
    if mask_air.sum() >= 50:
        X_air = df_nosack.loc[mask_air, X_base.columns]
        y_air = df_nosack.loc[mask_air, 'air_yards'].astype(float)
        trX, teX, trY, teY = _safe_split(X_air, y_air, val_size, random_state, stratify_ok=False)
        m_air = _reg(cat, num, kind='gbr')
        m_air.fit(trX, trY)
        ypred = m_air.predict(teX) if len(teX) else np.array([])
        mae_air = float(mean_absolute_error(teY, ypred)) if len(teX) else None
        rmse_air = float(np.sqrt(mean_squared_error(teY, ypred))) if len(teX) else None
        joblib.dump(m_air, outdir/'pass_model_air.pkl')
    else:
        mae_air = None; rmse_air = None

    y_res = np.where(df_nosack['interception']==1, 'interception',
              np.where(df_nosack['complete_pass']==1, 'complete', 'incomplete'))
    if 'receiver_drop' in df_nosack.columns and df_nosack['receiver_drop'].nunique() > 1:
        y_res = np.where(df_nosack['receiver_drop']==1, 'drop', y_res)
    X_res = df_nosack[X_base.columns]
    trX, teX, trY, teY = _safe_split(X_res, pd.Series(y_res), val_size, random_state, stratify_ok=True)
    m_res = _clf(cat, num)
    m_res.fit(trX, trY)
    yhat = m_res.predict(teX) if len(teX) else np.array([])
    acc_res = float(accuracy_score(teY, yhat)) if len(teX) else None
    joblib.dump(m_res, outdir/'pass_model_result.pkl')

    # --- YAC regression on completed passes ---
    df_comp = df_nosack[df_nosack['complete_pass'] == 1].copy()
    if len(df_comp) >= 50:
        y_yac = pd.to_numeric(
            df_comp['yards_after_catch'] if 'yards_after_catch' in df_comp.columns else pd.Series(index=df_comp.index, dtype=float),
            errors='coerce'
        )
        y_yac = y_yac.where(~y_yac.isna(), (df_comp['yards_gained'] - df_comp['air_yards'].fillna(0)).astype(float))
        y_yac = y_yac.clip(lower=0)
        mask = y_yac.notna() & np.isfinite(y_yac.values)
        X_yac = df_comp.loc[mask, X_base.columns]
        y_yac = y_yac.loc[mask]
        if len(y_yac) >= 50:
            trX, teX, trY, teY = _safe_split(X_yac, y_yac, val_size, random_state, stratify_ok=False)
            m_yac = _reg(cat, num, kind='rf')
            m_yac.fit(trX, trY)
            ypred = m_yac.predict(teX) if len(teX) else np.array([])
            mae_yac = float(mean_absolute_error(teY, ypred)) if len(teX) else None
            rmse_yac = float(np.sqrt(mean_squared_error(teY, ypred))) if len(teX) else None
            joblib.dump(m_yac, outdir/'pass_model_yac.pkl')
        else:
            mae_yac = None; rmse_yac = None
    else:
        mae_yac = None; rmse_yac = None

    df_comp_f = df_comp.copy()
    if 'fumble' in df_comp_f.columns:
        y_f = df_comp_f['fumble'].fillna(0).astype(int)
        trX, teX, trY, teY = _safe_split(df_comp_f[X_base.columns], y_f, val_size, random_state, stratify_ok=True)
        m_f = _clf(cat, num)
        m_f.fit(trX, trY)
        yhat = m_f.predict(teX) if len(teX) else np.array([])
        acc_f = float(accuracy_score(teY, yhat)) if len(teX) else None
        joblib.dump(m_f, outdir/'pass_model_rec_fumble.pkl')
    else:
        acc_f = None

    acc_fl = None
    if 'fumble_lost' in df_comp_f.columns and df_comp_f['fumble'].sum() >= 20 and df_comp_f['fumble_lost'].nunique() >= 2:
        df_f = df_comp_f[df_comp_f['fumble']==1]
        y_fl = df_f['fumble_lost'].fillna(0).astype(int)
        trX, teX, trY, teY = _safe_split(df_f[X_base.columns], y_fl, val_size, random_state, stratify_ok=True)
        m_fl = _clf(cat, num)
        m_fl.fit(trX, trY)
        yhat = m_fl.predict(teX) if len(teX) else np.array([])
        acc_fl = float(accuracy_score(teY, yhat)) if len(teX) else None
        joblib.dump(m_fl, outdir/'pass_model_rec_fumble_lost.pkl')

    meta = {
        "features_base": base,
        "categorical": cat,
        "labels_result": sorted(pd.unique(pd.Series(y_res))),
        "metrics": {
            "sack_acc": acc_sack,
            "air_mae": mae_air,
            "air_rmse": rmse_air,
            "result_acc": acc_res,
            "yac_mae": mae_yac,
            "yac_rmse": rmse_yac,
            "rec_fumble_acc": acc_f,
            "rec_fumble_lost_acc": acc_fl
        }
    }
    with open(outdir/'pass_models_meta.json','w') as f:
        json.dump(meta, f, indent=2)
    return meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=str, required=True)
    ap.add_argument('--outdir', type=str, default='.')
    ap.add_argument('--val_size', type=float, default=0.2)
    ap.add_argument('--random_state', type=int, default=42)
    args = ap.parse_args()
    df = load_engineer(args.csv)
    meta = train_all(df, Path(args.outdir), args.val_size, args.random_state)
    print(json.dumps(meta["metrics"], indent=2))

if __name__ == '__main__':
    main()
