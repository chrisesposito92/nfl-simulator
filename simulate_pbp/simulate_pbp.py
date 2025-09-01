import numpy as np
import pandas as pd
from dataclasses import dataclass

PATH = "../nfl_data_py/pbp_data.csv"

@dataclass
class Play:
    down: int
    yards_to_go: int
    play_type: str
    result: str
    yards_gained: int

@dataclass
class GameStats:
    home_points: int
    home_passing_yards: int
    home_passing_tds: int
    home_rushing_yards: int
    home_rushing_tds: int
    away_points: int
    away_passing_yards: int
    away_passing_tds: int
    away_rushing_yards: int
    away_rushing_tds: int

def determine_play_type(team, down, yards_to_go, yard_line, under_2_minutes):
    if down == 4:
        if yard_line > 75:
            return 'field_goal'
        else:
            return 'punt'
    pbp_data = pd.read_csv(PATH)
    pbp_data = pbp_data[pbp_data['home_team'] == team | pbp_data['away_team'] == team]
    # Find total number of rows where 'play_type' is 'pass'
    num_pass_plays = pbp_data[pbp_data['play_type'] == 'pass'].shape[0]
    # Find total number of rows where 'play_type' is 'run'
    num_run_plays = pbp_data[pbp_data['play_type'] == 'run'].shape[0]

def simulate_game(awayTeam, homeTeam):
    # Load play-by-play data
    pbp_data = pd.read_csv(PATH)

    # Start a list of Plays
    plays = []

    # Declare class variables
    game_is_over = False
    current_quarter = 1
    time_left_in_quarter = 900  # 15 minutes in seconds
    yards_to_go = 10
    current_down = 1
    current_yard_line = 25  # Starting at own 25 yard line
    quarter_is_over = False
    team_on_offense = homeTeam
    under_2_minutes = False

    # Home team stats
    game_stats = GameStats(
        home_points=0,
        home_passing_yards=0,
        home_passing_tds=0,
        home_rushing_yards=0,
        home_rushing_tds=0,
        away_points=0,
        away_passing_yards=0,
        away_passing_tds=0,
        away_rushing_yards=0,
        away_rushing_tds=0
    )
    first_quarter_stats = GameStats(
        home_points=0,
        home_passing_yards=0,
        home_passing_tds=0,
        home_rushing_yards=0,
        home_rushing_tds=0,
        away_points=0,
        away_passing_yards=0,
        away_passing_tds=0,
        away_rushing_yards=0,
        away_rushing_tds=0
    )
    second_quarter_stats = GameStats(
        home_points=0,
        home_passing_yards=0,
        home_passing_tds=0,
        home_rushing_yards=0,
        home_rushing_tds=0,
        away_points=0,
        away_passing_yards=0,
        away_passing_tds=0,
        away_rushing_yards=0,
        away_rushing_tds=0
    )
    third_quarter_stats = GameStats(
        home_points=0,
        home_passing_yards=0,
        home_passing_tds=0,
        home_rushing_yards=0,
        home_rushing_tds=0,
        away_points=0,
        away_passing_yards=0,
        away_passing_tds=0,
        away_rushing_yards=0,
        away_rushing_tds=0
    )
    fourth_quarter_stats = GameStats(
        home_points=0,
        home_passing_yards=0,
        home_passing_tds=0,
        home_rushing_yards=0,
        home_rushing_tds=0,
        away_points=0,
        away_passing_yards=0,
        away_passing_tds=0,
        away_rushing_yards=0,
        away_rushing_tds=0
    )

    # Simulate coin toss. For now, winner of coin toss gets ball first
    coin_toss = np.random.rand()
    if coin_toss < 0.5:
        team_on_offense = awayTeam

    while not quarter_is_over:
       # Determine play type
       play_type = determine_play_type(team_on_offense, current_down, yards_to_go, current_yard_line, under_2_minutes)

    return plays