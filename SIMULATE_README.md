# NFL Monte Carlo v0 to v1 (Defense + Pace)

## Goal
Simulate per team game outcomes:
- Passing yards and passing TDs
- Rushing yards and rushing TDs

## Data
Single weekly player file for 2020–2024 REG.
Fields used:
`recent_team, opponent_team, season, week, season_type, attempts, carries, passing_yards, passing_tds, rushing_yards, rushing_tds`

## Pipeline

### 1) Aggregate team weeks (offense)
**What**  
Sum player stats by `recent_team, season, week`.

**Why**  
Team level per game is the modeling unit.

**How**  
- `passing_yards_g = sum(passing_yards)`  
- `rushing_yards_g = sum(rushing_yards)`  
- `passing_tds_g = sum(passing_tds)`  
- `rushing_tds_g = sum(rushing_tds)`  
- `attempts_g = sum(attempts)`  
- `carries_g = sum(carries)`  
- `plays_g = attempts_g + carries_g`

### 2) Aggregate defense weeks (allowed)
**What**  
Sum what offenses produced vs each `opponent_team`. Rename to `team`.

**Why**  
That is defense allowed per game.

**How**  
Same sums as offense. Plus `plays_allowed_g = attempts + carries`.

### 3) Fit offense per play baselines
**What**  
Convert offense to per play rates and average volume.

**Why**  
Lets pace and defense scale cleanly.

**How** over the chosen window  
- Per play yardage  
  - `pass_ypp = sum(passing_yards) / sum(attempts)`  
  - `rush_ypp = sum(rushing_yards) / sum(carries)`  
- Per play TD rates  
  - `pass_tdr = sum(passing_tds) / sum(attempts)`  
  - `rush_tdr = sum(rushing_tds) / sum(carries)`  
- Expected volumes  
  - `E_att = mean(attempts_g)`  
  - `E_car = mean(carries_g)`  
- Dispersion anchors for yards  
  - `sigma_pass = stdev(passing_yards_g)` with floor 5  
  - `sigma_rush = stdev(rushing_yards_g)` with floor 5

### 4) Fit defense multipliers
**What**  
League relative strength on a per play basis and volume allowed.

**Why**  
Scale the opposing offense to the matchup.

**How**  
- League per play baselines: `L_pass_ypp, L_rush_ypp, L_pass_tdr, L_rush_tdr`  
- Defense allowed per play: `D_pass_ypp, D_rush_ypp, D_pass_tdr, D_rush_tdr`  
- Multipliers  
  - `m_ypp_pass = D_pass_ypp / L_pass_ypp`  
  - `m_ypp_rush = D_rush_ypp / L_rush_ypp`  
  - `m_tdr_pass = D_pass_tdr / L_pass_tdr`  
  - `m_tdr_rush = D_rush_tdr / L_rush_tdr`  
- Pace multipliers  
  - `m_att = mean(att_allowed) / league_mean(att)`  
  - `m_car = mean(car_allowed) / league_mean(car)`

### 5) Build matchup adjusted offense
**What**  
Turn per play rates and pace into per game parameters.

**Why**  
These feed the simulation.

**How**  
- Expected volumes vs defense  
  - `E_att' = E_att * m_att`  
  - `E_car' = E_car * m_car`  
- Adjusted per play rates  
  - `pass_ypp' = pass_ypp * m_ypp_pass`  
  - `rush_ypp' = rush_ypp * m_ypp_rush`  
  - `pass_tdr' = pass_tdr * m_tdr_pass`  
  - `rush_tdr' = rush_tdr * m_tdr_rush`  
- Per game expectations  
  - `mu_pass_yards = pass_ypp' * E_att'`  
  - `mu_rush_yards = rush_ypp' * E_car'`  
  - `lambda_pass_td = pass_tdr' * E_att'`  
  - `lambda_rush_td = rush_tdr' * E_car'`

### 6) Sigma scaling modes
**What**  
How yardage sigma responds to pace and defense.

**Why**  
Variance should move with opportunity.

**How**  
- `none`: keep baseline sigmas  
- `proportional`: scale sigma by ratio of new mu to baseline mu  
- `sqrt_volume`: scale sigma by square root of volume ratio

### 7) Simulation
**What**  
Draw outcomes for both teams independently.

**Why**  
Get a distribution for game results.

**How**  
- Yards ~ Normal(`mu`, `sigma`), truncate at 0, round to int  
- TDs ~ Poisson(`lambda`)  
- Return draws and a quantile summary

## Assumptions
- Teams independent.  
- Pass and rush independent within a team.  
- TD counts are Poisson.  
- Yardage is normal with fixed or scaled sigma.  
- Stationary inside the sample window unless using `last_n`.

## Controls
- `season_min, season_max, season_type`  
- `last_n_off, last_n_def` for recency  
- `sigma_mode in {"none","proportional","sqrt_volume"}`  
- `n_sims` and random seed

## Sanity checks
- Compare simulated and actual histograms for yards and TDs.  
- Check TD overdispersion.  
- Calibration of p10 to p90 coverage.  
- Optional clamp for multipliers to `[0.6, 1.6]`.

## Incremental roadmap
1. Home and away splits for offense and defense.  
2. Weighted recency with exponential decay.  
3. Pass rate modeling. Sample pass share. Split plays by beta draw.  
4. Negative binomial for TDs if overdispersed.  
5. Environment. Dome flag. Wind and temperature bins as simple multipliers.  
6. Schedule normalization with iterative shrinkage to league.  
7. QB availability gating by games started. Shrink when small sample.  
8. Correlation layer via a shared game script shock.  
9. Red zone efficiency splits when play by play is added.  
10. Regularization toward league means based on sample size.

## Current modules
- `aggregate_teamweeks`  
- `aggregate_defenseweeks`  
- `fit_offense_perplay`  
- `fit_defense_adj`  
- `adjust_offense`  
- `_simulate_team`  
- `simulate_matchup_with_pace`  
- `summarize`

## Glossary
- `ypp`: yards per play  
- `tdr`: touchdowns per play rate  
- `E_att, E_car`: expected attempts and carries  
- `mu`: mean for yards  
- `lambda`: mean rate for TD counts