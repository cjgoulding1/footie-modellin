# footie-modellin

A model for predicting **how the first goal of a football match is scored**
- `shot`, `header`, `penalty`, `free_kick`, `own_goal`, or `no_goal` - and
backtesting it against bookmaker odds for these exotic markets.

Raw data (scraped shots, football-data.co.uk CSVs, the SQLite DB, and
generated backtest logs/outputs) is kept **local only** and is not checked
into this repo (see `.gitignore`). Everything here is the modeling code.

## Design

The full design rationale lives in
[`first_goal_model/PLAN.md`](first_goal_model/PLAN.md) - read that first for
the "why". Summary:

- **Stage 1 - who scores, if anyone?** Rather than fitting team strengths
  from historical goals, each match's own pre-match 1X2 and Over/Under 2.5
  odds are de-vigged and inverted into that match's Dixon-Coles parameters
  (`lambda_home`, `lambda_away`, `rho`) via a per-match least-squares solve
  (`dixon_coles.py`, `odds.py`). This gives `P(no_goal)` and `P(team scores
  first)` directly from the market, with no staleness/decay question since
  the odds are already point-in-time.

- **Stage 2 - if a team scores, by what method?** Needs team-specific
  history, not just the market price:
  - **Header** (`header_strength.py`): Dixon-Coles-style attack x defense
    fit - `lambda_header(i) = lambda_total(i) * c * a_i * b_j * calib` -
    fit via decayed quasi-Poisson MLE on headed xG, with an L2 penalty
    toward "average team" for regularization/identifiability, plus a
    post-hoc calibration factor correcting for headers underperforming
    their raw xG (~0.84-0.91x in this data).
  - **Penalty - two competing models**:
    - **Model A** (`method_shares.py`): empirical-Bayes shrunk share of a
      team's own goals that are penalties, decay-weighted, blended toward
      the league average.
    - **Model B** (`penalty_model.py`): per-league logistic regression,
      `P(team awarded >=1 penalty) = sigmoid(b0 + b1*lambda_own +
      b2*lambda_opp + b3*draw_rate)`, converted to a rate via
      `lambda_penalty = P(awarded) * conversion_rate` (small-probability
      Poisson approximation, `P(X>=1) ~= lambda`).
  - **Free kick / own goal**: plain league-average constants - team-level
    variance here turned out to be personnel-driven (free kicks) or just a
    proxy for attack volume (own goals), not a durable team trait.
  - **Shot**: the residual after the other four, not modeled directly
    (~70% of all goals).

- **Frozen-constants variant** (`train_frozen_constants.py`,
  `full_model_frozen.py`): instead of refitting the header/penalty models
  for every match (what `populate_model_columns.py` / `full_model.py` do
  live), fit each ONCE on a fixed training window (seasons 2016-2018) and
  freeze the result (`header_constants_2016_2018.json`,
  `penalty_constants_2016_2018.json`), then apply those fixed constants
  unchanged to seasons 2018/2019-present. This is what the backtests below
  run against. La_liga's penalty model uses a simpler 3-feature variant
  (drops `draw_rate`) - its 4-feature fit on this window was unstable
  (sign-flipped `draw_rate` coefficient, an artifact of the small ~2,280-row
  training sample, not a real effect).

## Backtesting results

All backtests use Kelly staking and the same Bet365-style dynamic pricing
rule (price depends on `avg_under_2_5`), discovered by inspection:

| Market  | `avg_under_2_5 <= 2.00` | `avg_under_2_5 > 2.00` |
|---------|------------------------:|-----------------------:|
| shot    | 1.50                    | 1.44                   |
| header  | 6.50                    | 6.00                   |
| penalty | 7.50                    | 7.00                   |

**Frozen constants (trained on seasons 2016-2018), tested on seasons
2018-present, all 5 leagues, 14,282 matches**
(`backtest_full_model_2018_dynamic.py`). Note season 2018 partially
overlaps the training window, so that one season isn't strictly
out-of-sample for header/penalty.

| Market | Kelly fraction | Min edge | Bets placed | Hit rate | Avg model prob | Final bankroll (from 100) | Return | Max drawdown |
|---|---|---|---|---|---|---|---|---|
| Shot | half | 0 | 1,778 (12.4%) | 72.0% | 70.71% | 393.51 | **+293.51%** | 49.19% |
| Header (uncapped) | half | 0 | 5,119 (35.8%) | 16.7% | 17.19% | 1,279.63 | +1,179.63% | 72.66% |
| Header (tuned) | quarter | 3pp | 469 (3.3%) | 19.2% | 19.45% | 287.90 | **+187.90%** | 31.21% |
| Penalty | half | 0 | 0 (0%) | - | - | 100.00 | 0% | 0% |

- **Shot** is the cleanest result: moderate bet frequency, model
  probability tracking the realized hit rate closely (70.71% vs 72.0%),
  solid compounding return, drawdown consistent with shorter-window runs.
- **Header (uncapped)** posts a huge headline return, but its own stated
  probability (17.19%) is slightly *above* its actual hit rate (16.7%) -
  the model is marginally overconfident - and it bets on over a third of
  all matches. Combined with a 72.66% max drawdown, that's the signature
  of half-Kelly staking too aggressively on a thin, noisy edge.
- **Header (tuned)**: quarter-Kelly + a minimum 3-percentage-point edge
  requirement cuts bet frequency ~11x (5,119 -> 469), nearly eliminates the
  overconfidence gap (19.45% vs 19.2%), and roughly halves max drawdown
  (72.66% -> 31.21%), at the cost of a much smaller (but still strong)
  compounding return. This is the safer operating point.
- **Penalty** never bets at the given prices (7.0/7.5): fair odds there
  imply ~13-14% probability, well above this dataset's actual penalty-as-
  first-goal rate (~7.7%), so the model never finds a positive edge against
  those specific prices. Not a bug - the model is correctly refusing
  action it doesn't believe in.
- **Running shot and header as two independent bankrolls** (100 each):
  combined profit +481.41 on 200 staked, **+240.7%** overall, with each
  market's drawdown staying contained to its own pot rather than
  compounding into the other's.

Earlier (non-frozen, per-match live-refit) backtest for comparison, shot +
header only, seasons 2024-2025 (`backtest_full_model.py`): shot +65.18%
return / 12.75% drawdown; header +714.42% return / 53.33% drawdown. The
frozen-constants approach generalizes the header edge further out of
sample (more bets, longer window) but at a higher realized drawdown before
tuning.

## File guide (`first_goal_model/`)

| File | Role |
|---|---|
| `PLAN.md` | Design doc - read first |
| `db.py` | SQLite connection helper |
| `dixon_coles.py` | tau-adjusted bivariate Poisson + per-match (lambda, rho) solver |
| `odds.py` | De-vig 1X2 / O-U 2.5 market odds |
| `labels.py` | Ground-truth first-goal-method labels from raw shot data |
| `header_strength.py` | Header attack/defense MLE fit |
| `select_xi.py` | Cross-validated decay half-life for the header model |
| `method_shares.py` | Penalty model A (shrunk share) + free_kick/own_goal league constants |
| `penalty_model.py` | Penalty model B (logistic regression) |
| `select_penalty_xi.py`, `select_penalty_regularization.py`, `evaluate_penalty_model.py` | Penalty model B tuning/evaluation |
| `populate_model_columns.py` | Populates live per-match Dixon-Coles/header columns on `matches` |
| `full_model.py` | Full 6-way model, live per-match refit |
| `train_frozen_constants.py` | Fits header/penalty constants once on seasons 2016-2018 |
| `full_model_frozen.py` | Full 6-way model using the frozen constants |
| `header_constants_2016_2018.json`, `penalty_constants_2016_2018.json` | The frozen constants themselves |
| `backtest_full_model.py` | Dynamic shot+header backtest, live refit, seasons 2024-2025 |
| `backtest_full_model_2019.py` | Shot-only backtest, frozen constants, seasons 2019-present |
| `backtest_full_model_2018_dynamic.py` | Shot+header+penalty backtest, frozen constants, seasons 2018-present, per-market Kelly fraction/min-edge tuning |

Root-level scraping/ingestion scripts (`scrape_understat.py`,
`import_e0_csv.py`) build the local-only SQLite DB and aren't part of this
push.
