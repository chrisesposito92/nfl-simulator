python simulate_pbp/train_gated_play_models.py --csv nfl_data_py/pbp_data.csv --outdir simulate_pbp/models --fg_max 70

python simulate_pbp/train_punt_models.py --csv nfl_data_py/pbp_data.csv --outdir simulate_pbp --val_size 0.2 --random_state 42

python simulate_pbp/train_run_models.py --csv nfl_data_py/pbp_data.csv --outdir simulate_pbp --val_size 0.2 --random_state 42

python simulate_pbp/train_pass_models.py --csv nfl_data_py/pbp_data.csv --outdir simulate_pbp --val_size 0.2 --random_state 42

python simulate_pbp/train_gated_play_models.py \
  --csv nfl_data_py/pbp_data.csv \
  --outdir simulate_pbp/models \
  --fg_max 70 \
  --val_size 0.2 \
  --random_state 42


# deterministic (current behavior)
predict_play_type('BUF', 1, 10, 75, 2, 0, 0, 3, 3, decision='argmax')

# probabilistic sampling
predict_play_type('BUF', 1, 10, 75, 2, 0, 0, 3, 3, decision='sample', temperature=1.0, random_state=42)

# epsilon-greedy: mostly argmax, sometimes explore
predict_play_type('BUF', 1, 10, 75, 2, 0, 0, 3, 3, decision='epsilon_greedy', epsilon=0.1, random_state=42)

python -c "import sys; sys.path.append('simulate_pbp'); from punt_sim import simulate_punt; print(simulate_punt('BUF','NE',10,70,2,0,0,3,3, decision='sample'))"

python -c "import sys; sys.path.append('simulate_pbp'); from run_sim import simulate_run; \
print(simulate_run('BUF','NE',1,10,75,2,0,0,3,3, decision='sample', random_state=42))"

python -c "import sys; sys.path.append('simulate_pbp'); from pass_sim import simulate_pass; \
print(simulate_pass('BUF','NE',1,10,75,2,0,0,3,3, decision='sample', random_state=42))"