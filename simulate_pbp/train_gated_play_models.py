import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
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
        return int(m) * 60 + int(s)
    except Exception:
        return np.nan

def load_engineer(csv_path: str):
    df = pd.read_csv(csv_path, low_memory=False)

    cmap = {}
    cmap['posteam'] = _col(df, ['posteam','offense_team','offense','OffenseTeam'])
    cmap['defteam'] = _col(df, ['defteam','defense_team','defense','DefenseTeam'])
    cmap['down'] = _col(df, ['down','Down'])
    cmap['ydstogo'] = _col(df, ['ydstogo','YardsToGo','yds_to_go'])
    cmap['yardline_100'] = _col(df, ['yardline_100','YardLine_100','yard_line_100','yardline100'])
    cmap['qtr'] = _col(df, ['qtr','quarter','Quarter'])
    cmap['play_type'] = _col(df, ['play_type','PlayType','playType'])
    cmap['quarter_seconds_remaining'] = _col(df, ['quarter_seconds_remaining','QuarterSecondsRemaining'])
    cmap['clock'] = _col(df, ['clock','Clock'])
    cmap['score_differential'] = _col(df, ['score_differential','ScoreDiff','scoreDiff'])
    cmap['posteam_timeouts_remaining'] = _col(df, ['posteam_timeouts_remaining','posteam_timeouts','OffenseTimeouts','offense_timeouts_remaining'])
    cmap['defteam_timeouts_remaining'] = _col(df, ['defteam_timeouts_remaining','defteam_timeouts','DefenseTimeouts','defense_timeouts_remaining'])

    needed = ['posteam','down','ydstogo','yardline_100','qtr','play_type']
    miss = [k for k in needed if cmap.get(k) is None]
    if miss:
        raise RuntimeError(f"Missing columns: {miss}")

    # keep only modeled labels
    df = df[df[cmap['play_type']].isin(['run','pass','punt','field_goal'])].copy()

    # derive quarter_seconds_remaining if needed
    if cmap['quarter_seconds_remaining'] is None:
        df['quarter_seconds_remaining'] = df[cmap['clock']].map(_parse_clock) if cmap['clock'] else np.nan
    elif cmap['quarter_seconds_remaining'] != 'quarter_seconds_remaining':
        df = df.rename(columns={cmap['quarter_seconds_remaining']: 'quarter_seconds_remaining'})

    # under 2 minutes in Q2 or Q4
    df['under_2_minutes'] = np.where(
        (df['quarter_seconds_remaining'] <= 120) & (df[cmap['qtr']].isin([2, 4])),
        1, 0
    )

    # normalize column names
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

    ren = {}
    for k in ['posteam','defteam','down','ydstogo','yardline_100','qtr','play_type']:
        if cmap[k] != k:
            ren[cmap[k]] = k
    if ren:
        df = df.rename(columns=ren)

    # types
    df = df.dropna(subset=['posteam','down','ydstogo','yardline_100','qtr','play_type'])
    df['down'] = df['down'].astype(int)
    df['ydstogo'] = df['ydstogo'].astype(float)
    df['yardline_100'] = df['yardline_100'].astype(float)
    df['qtr'] = df['qtr'].astype(int)
    df['under_2_minutes'] = df['under_2_minutes'].fillna(0).astype(int)
    df['score_differential'] = df['score_differential'].astype(float)
    df['posteam_timeouts_remaining'] = df['posteam_timeouts_remaining'].astype(int)
    df['defteam_timeouts_remaining'] = df['defteam_timeouts_remaining'].astype(int)

    # FG distance heuristic
    df['fg_distance'] = df['yardline_100'] + 17.0
    return df

def make_pipeline(cat_cols, num_cols):
    pre = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', 'passthrough', num_cols)
    ])
    clf = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    return Pipeline([('pre', pre), ('clf', clf)])

def split_data(df, y_col, val_size=0.0, random_state=42, stratify=True):
    if val_size and val_size > 0.0 and len(df):
        if stratify and df[y_col].nunique() > 1:
            tr, te = train_test_split(df, test_size=val_size, random_state=random_state, stratify=df[y_col])
        else:
            tr, te = train_test_split(df, test_size=val_size, random_state=random_state)
        return tr, te
    return df, df.iloc[0:0]

def train_all(df, fg_max=70.0, outdir=Path("."), val_size=0.0, random_state=42):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    base_feats = [
        'posteam','down','ydstogo','yardline_100','qtr',
        'under_2_minutes','score_differential',
        'posteam_timeouts_remaining','defteam_timeouts_remaining'
    ]
    cat = ['posteam','qtr']

    # 1st–3rd: pass vs run
    early = df[(df['down'].between(1, 3)) & (df['play_type'].isin(['pass', 'run']))].copy()
    train_df, test_df = split_data(early, 'play_type', val_size, random_state)
    pipe_pr = make_pipeline(cat, [c for c in base_feats if c not in cat])
    pipe_pr.fit(train_df[base_feats], train_df['play_type'])
    yhat = pipe_pr.predict(test_df[base_feats]) if len(test_df) else []
    acc_pr = float(accuracy_score(test_df['play_type'], yhat)) if len(test_df) else None
    joblib.dump(pipe_pr, outdir / 'model_early_pass_run.pkl')

    # 4th down gating
    fourth = df[df['down'] == 4].copy()
    fourth['kick_window'] = fourth['fg_distance'] <= fg_max

    # 4th, FG window: field_goal vs punt vs go
    kick3 = fourth[fourth['kick_window']].copy()
    if len(kick3):
        kick3['kick3_target'] = np.where(kick3['play_type'].isin(['punt', 'field_goal']), kick3['play_type'], 'go')
        feats4 = base_feats + ['fg_distance']
        train_df, test_df = split_data(kick3, 'kick3_target', val_size, random_state)
        pipe_k3 = make_pipeline(cat, [c for c in feats4 if c not in cat])
        pipe_k3.fit(train_df[feats4], train_df['kick3_target'])
        yhat = pipe_k3.predict(test_df[feats4]) if len(test_df) else []
        acc_k3 = float(accuracy_score(test_df['kick3_target'], yhat)) if len(test_df) else None
        joblib.dump(pipe_k3, outdir / 'model_4th_kick3.pkl')
    else:
        acc_k3 = None

    # 4th, no FG window: punt vs go
    nofg = fourth[~fourth['kick_window']].copy()
    if len(nofg):
        nofg['pg_target'] = np.where(nofg['play_type'].isin(['punt']), 'punt', 'go')
        feats4 = base_feats + ['fg_distance']
        train_df, test_df = split_data(nofg, 'pg_target', val_size, random_state)
        pipe_pg = make_pipeline(cat, [c for c in feats4 if c not in cat])
        pipe_pg.fit(train_df[feats4], train_df['pg_target'])
        yhat = pipe_pg.predict(test_df[feats4]) if len(test_df) else []
        acc_pg = float(accuracy_score(test_df['pg_target'], yhat)) if len(test_df) else None
        joblib.dump(pipe_pg, outdir / 'model_4th_punt_go.pkl')
    else:
        acc_pg = None

    # 4th, going: pass vs run
    go4 = fourth[fourth['play_type'].isin(['pass', 'run'])].copy()
    if len(go4):
        train_df, test_df = split_data(go4, 'play_type', val_size, random_state)
        pipe_pr4 = make_pipeline(cat, [c for c in base_feats if c not in cat] + ['fg_distance'])
        pipe_pr4.fit(train_df[base_feats + ['fg_distance']], train_df['play_type'])
        yhat = pipe_pr4.predict(test_df[base_feats + ['fg_distance']]) if len(test_df) else []
        acc_pr4 = float(accuracy_score(test_df['play_type'], yhat)) if len(test_df) else None
        joblib.dump(pipe_pr4, outdir / 'model_4th_pass_run.pkl')
    else:
        acc_pr4 = None

    metrics = {
        'early_pass_run_val_acc': acc_pr,
        'fourth_kick3_val_acc': acc_k3,
        'fourth_punt_go_val_acc': acc_pg,
        'fourth_pass_run_val_acc': acc_pr4
    }
    with open(outdir / 'gated_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    return metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=str, required=True)
    ap.add_argument('--outdir', type=str, default='.')
    ap.add_argument('--fg_max', type=float, default=70.0)
    ap.add_argument('--val_size', type=float, default=0.0)  # 0.0 uses all data
    ap.add_argument('--random_state', type=int, default=42)
    args = ap.parse_args()
    df = load_engineer(args.csv)
    metrics = train_all(
        df,
        fg_max=args.fg_max,
        outdir=Path(args.outdir),
        val_size=args.val_size,
        random_state=args.random_state
    )
    print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    main()