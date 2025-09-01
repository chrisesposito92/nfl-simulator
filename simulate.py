import numpy as np
import pandas as pd
from dataclasses import dataclass

PATH = "nfl_data_py/weekly_data.csv"

@dataclass
class OffensePerPlay:
    pass_ypp: float
    rush_ypp: float
    pass_tdr: float
    rush_tdr: float
    att_mu: float
    carry_mu: float
    pass_yards_sigma: float
    rush_yards_sigma: float

@dataclass
class DefenseAdj:
    m_ypp_pass: float
    m_ypp_rush: float
    m_tdr_pass: float
    m_tdr_rush: float
    m_att: float
    m_carry: float

@dataclass
class OffenseParams:
    pass_yards_mu: float
    pass_yards_sigma: float
    rush_yards_mu: float
    rush_yards_sigma: float
    pass_td_lambda: float
    rush_td_lambda: float

def _safe_sigma(v):
    v = np.asarray(v, float)
    if v.size <= 1:
        return 5.0
    s = float(np.std(v, ddof=1))
    return s if s > 1e-9 else 5.0

def aggregate_teamweeks(df_players, season_min=2020, season_max=2024, season_type="REG"):
    df = df_players.copy()
    for c in ["passing_yards","passing_tds","rushing_yards","rushing_tds","attempts","carries"]:
        df[c] = df[c].fillna(0)
    df = df[(df["season"] >= season_min) & (df["season"] <= season_max) & (df["season_type"] == season_type)]
    g = (df.groupby(["recent_team","season","week"], as_index=False)
           [["passing_yards","passing_tds","rushing_yards","rushing_tds","attempts","carries"]]
           .sum())
    g = g.sort_values(["recent_team","season","week"]).reset_index(drop=True)
    g["plays"] = g["attempts"] + g["carries"]
    return g

def aggregate_defenseweeks(df_players, season_min=2020, season_max=2024, season_type="REG"):
    df = df_players.copy()
    for c in ["passing_yards","passing_tds","rushing_yards","rushing_tds","attempts","carries"]:
        df[c] = df[c].fillna(0)
    df = df[(df["season"] >= season_min) & (df["season"] <= season_max) & (df["season_type"] == season_type)]
    g = (df.groupby(["opponent_team","season","week"], as_index=False)
           [["passing_yards","passing_tds","rushing_yards","rushing_tds","attempts","carries"]]
           .sum()).rename(columns={"opponent_team":"team"})
    g = g.sort_values(["team","season","week"]).reset_index(drop=True)
    g["plays_allowed"] = g["attempts"] + g["carries"]
    return g

def fit_offense_perplay(teamweeks, team, last_n=None) -> OffensePerPlay:
    x = teamweeks[teamweeks["recent_team"] == team].sort_values(["season","week"])
    if last_n and last_n > 0:
        x = x.tail(last_n)
    att_tot = float(x["attempts"].sum())
    car_tot = float(x["carries"].sum())
    pass_ypp = float(x["passing_yards"].sum()) / att_tot if att_tot > 0 else 0.0
    rush_ypp = float(x["rushing_yards"].sum()) / car_tot if car_tot > 0 else 0.0
    pass_tdr = float(x["passing_tds"].sum()) / att_tot if att_tot > 0 else 0.0
    rush_tdr = float(x["rushing_tds"].sum()) / car_tot if car_tot > 0 else 0.0
    att_mu = float(x["attempts"].mean()) if len(x) else 0.0
    carry_mu = float(x["carries"].mean()) if len(x) else 0.0
    sig_p = _safe_sigma(x["passing_yards"].to_numpy())
    sig_r = _safe_sigma(x["rushing_yards"].to_numpy())
    return OffensePerPlay(pass_ypp, rush_ypp, pass_tdr, rush_tdr, att_mu, carry_mu, sig_p, sig_r)

def fit_defense_adj(defenseweeks, team, last_n=None) -> DefenseAdj:
    x = defenseweeks[defenseweeks["team"] == team].sort_values(["season","week"])
    if last_n and last_n > 0:
        x = x.tail(last_n)
    league = defenseweeks
    att_tot_l = float(league["attempts"].sum())
    car_tot_l = float(league["carries"].sum())
    pypp_l = float(league["passing_yards"].sum()) / att_tot_l if att_tot_l > 0 else 0.0
    rypp_l = float(league["rushing_yards"].sum()) / car_tot_l if car_tot_l > 0 else 0.0
    ptdr_l = float(league["passing_tds"].sum()) / att_tot_l if att_tot_l > 0 else 0.0
    rtdr_l = float(league["rushing_tds"].sum()) / car_tot_l if car_tot_l > 0 else 0.0

    att_tot = float(x["attempts"].sum())
    car_tot = float(x["carries"].sum())
    pypp_d = float(x["passing_yards"].sum()) / att_tot if att_tot > 0 else 0.0
    rypp_d = float(x["rushing_yards"].sum()) / car_tot if car_tot > 0 else 0.0
    ptdr_d = float(x["passing_tds"].sum()) / att_tot if att_tot > 0 else 0.0
    rtdr_d = float(x["rushing_tds"].sum()) / car_tot if car_tot > 0 else 0.0

    m_ypp_pass = (pypp_d / pypp_l) if pypp_l > 0 else 1.0
    m_ypp_rush = (rypp_d / rypp_l) if rypp_l > 0 else 1.0
    m_tdr_pass = (ptdr_d / ptdr_l) if ptdr_l > 0 else 1.0
    m_tdr_rush = (rtdr_d / rtdr_l) if rtdr_l > 0 else 1.0

    att_mu_l = float(league["attempts"].mean()) if len(league) else 0.0
    car_mu_l = float(league["carries"].mean()) if len(league) else 0.0
    att_mu_d = float(x["attempts"].mean()) if len(x) else 0.0
    car_mu_d = float(x["carries"].mean()) if len(x) else 0.0
    m_att = (att_mu_d / att_mu_l) if att_mu_l > 0 else 1.0
    m_carry = (car_mu_d / car_mu_l) if car_mu_l > 0 else 1.0

    return DefenseAdj(m_ypp_pass, m_ypp_rush, m_tdr_pass, m_tdr_rush, m_att, m_carry)

def adjust_offense(off: OffensePerPlay, d: DefenseAdj, sigma_mode="none") -> OffenseParams:
    E_att = off.att_mu * d.m_att
    E_carry = off.carry_mu * d.m_carry
    pass_ypp = off.pass_ypp * d.m_ypp_pass
    rush_ypp = off.rush_ypp * d.m_ypp_rush
    pass_tdr = off.pass_tdr * d.m_tdr_pass
    rush_tdr = off.rush_tdr * d.m_tdr_rush
    mu_pass = pass_ypp * E_att
    mu_rush = rush_ypp * E_carry
    lam_pass = pass_tdr * E_att
    lam_rush = rush_tdr * E_carry
    if sigma_mode == "proportional":
        base_mu_pass = off.pass_ypp * off.att_mu
        base_mu_rush = off.rush_ypp * off.carry_mu
        s_p = (mu_pass / base_mu_pass) if base_mu_pass > 1e-9 else 1.0
        s_r = (mu_rush / base_mu_rush) if base_mu_rush > 1e-9 else 1.0
        sig_p = max(off.pass_yards_sigma * s_p, 5.0)
        sig_r = max(off.rush_yards_sigma * s_r, 5.0)
    elif sigma_mode == "sqrt_volume":
        s_p = np.sqrt(max(E_att,1.0)/max(off.att_mu,1.0))
        s_r = np.sqrt(max(E_carry,1.0)/max(off.carry_mu,1.0))
        sig_p = max(off.pass_yards_sigma * s_p, 5.0)
        sig_r = max(off.rush_yards_sigma * s_r, 5.0)
    else:
        sig_p = off.pass_yards_sigma
        sig_r = off.rush_yards_sigma
    return OffenseParams(mu_pass, sig_p, mu_rush, sig_r, lam_pass, lam_rush)

def _simulate_team(p: OffenseParams, rng, n):
    py = rng.normal(p.pass_yards_mu, p.pass_yards_sigma, size=n)
    ry = rng.normal(p.rush_yards_mu, p.rush_yards_sigma, size=n)
    py = np.clip(np.rint(py), 0, None).astype(int)
    ry = np.clip(np.rint(ry), 0, None).astype(int)
    ptd = rng.poisson(p.pass_td_lambda, size=n).astype(int)
    rtd = rng.poisson(p.rush_td_lambda, size=n).astype(int)
    return py, ptd, ry, rtd

def simulate_matchup_with_pace(teamA, teamB, players_df,
                               season_min=2020, season_max=2024, season_type="REG",
                               last_n_off=None, last_n_def=None,
                               n_sims=50000, seed=None, sigma_mode="none"):
    teamweeks = aggregate_teamweeks(players_df, season_min, season_max, season_type)
    defweeks = aggregate_defenseweeks(players_df, season_min, season_max, season_type)
    A_off = fit_offense_perplay(teamweeks, teamA, last_n_off)
    B_off = fit_offense_perplay(teamweeks, teamB, last_n_off)
    A_def = fit_defense_adj(defweeks, teamA, last_n_def)
    B_def = fit_defense_adj(defweeks, teamB, last_n_def)
    A_params = adjust_offense(A_off, B_def, sigma_mode=sigma_mode)
    B_params = adjust_offense(B_off, A_def, sigma_mode=sigma_mode)
    rng = np.random.default_rng(seed)
    A_py, A_ptd, A_ry, A_rtd = _simulate_team(A_params, rng, n_sims)
    B_py, B_ptd, B_ry, B_rtd = _simulate_team(B_params, rng, n_sims)
    sims = pd.DataFrame({
        "A_pass_yards": A_py, "A_pass_TDs": A_ptd, "A_rush_yards": A_ry, "A_rush_TDs": A_rtd,
        "B_pass_yards": B_py, "B_pass_TDs": B_ptd, "B_rush_yards": B_ry, "B_rush_TDs": B_rtd,
    })
    return sims, A_params, B_params

def summarize(df, qs=(0.1,0.5,0.9)):
    out = pd.DataFrame({"mean": df.mean(numeric_only=True)})
    for q in qs:
        out[f"p{int(q*100)}"] = df.quantile(q, numeric_only=True)
    return out

if __name__ == "__main__":
    players = pd.read_csv(PATH)
    sims, kc_adj, buf_adj = simulate_matchup_with_pace(
        "KC", "BUF", players,
        season_min=2020, season_max=2024, season_type="REG",
        last_n_off=None, last_n_def=None,  # set to 16 for recency
        n_sims=50000, seed=42, sigma_mode="proportional"  # "none" or "sqrt_volume" also valid
    )
    print("KC adjusted params:", kc_adj)
    print("BUF adjusted params:", buf_adj)
    print(summarize(sims).round(2).to_string())