# First Goal Model — Design Doc

## Goal

For a given fixture, predict the probability distribution over how the
**first goal of the match** happens:

- `shot` — a normal goal (any body part except head, not a penalty or direct
  free kick)
- `header`
- `penalty`
- `free_kick` — direct free kick
- `own_goal`
- `no_goal` — match finishes 0-0

These six outcomes are mutually exclusive and exhaustive, so the model's
output for a fixture must sum to 1.

## Why two stages

The problem splits naturally:

1. **Who scores, if anyone?** — This only needs each team's overall goal
   threat, which the betting market already prices very efficiently via 1X2
   and Over/Under odds. We use Dixon-Coles to turn those market odds into
   each team's expected goals (λ) for this specific fixture, then derive
   the probability of no goal at all, and the probability each side scores
   the game's first goal.
2. **If a team scores, what method?** — This is where team identity matters
   beyond raw goal threat: some teams score more of their goals via headers
   (aerial-focused attack, or weak in the air defensively), draw more
   penalties, etc. This needs team-specific history from the shot-level
   data, not just a market price.

Stage 1 answers "how many, and who first." Stage 2 answers "scored by what
method." Multiplying them together gives the full six-way distribution.

## Data audited so far (understat.db)

- `shots.result`: `Goal`, `OwnGoal`, `SavedShot`, `MissedShots`,
  `BlockedShot`, `ShotOnPost`. Own goals are a genuinely separate `result`
  value (1,763 across the dataset) and always carry `xg = 0` — understat
  doesn't credit a "shot" the scorer intended to take, so own goals can't be
  modeled via xG at all, only via historical frequency.
- `shots.situation`: `OpenPlay`, `FromCorner`, `SetPiece`, `Penalty`,
  `DirectFreekick`. Note `SetPiece` is a mixed bag of headers (1,331) and
  footed shots (1,866) off indirect set pieces — it is NOT a proxy for
  "header."
- `shots.shot_type`: `RightFoot`, `LeftFoot`, `Head`, `OtherBodyPart`.

**Classification rule for a goal event** (mutually exclusive, applied in
this order):

```
1. result == 'OwnGoal'          -> own_goal
2. situation == 'Penalty'       -> penalty
3. situation == 'DirectFreekick'-> free_kick
4. shot_type == 'Head'          -> header
5. else                         -> shot
```

Verified this partitions all 59,763 goal-scoring events (58,000 `Goal` +
1,763 `OwnGoal`) with no overlap and no leftover category.

**Findings that shaped the plan:**

- Free-kick scoring rate varies ~18x team to team (Barcelona 0.081/match vs
  Watford 0.0044/match) but this is overwhelmingly a personnel effect
  (Messi), not a durable team trait, and our data is match-level, not
  roster-level. Free kicks are modeled as **league-average, not
  team-specific**.
- Own-goal-benefit rate varies ~4x (PSG/Spurs/Arsenal ~0.075 vs
  Getafe/Augsburg ~0.02), but this looks like it's just proxying attacking
  pressure/volume rather than an independent skill. Own goals are modeled
  as a **fixed small fraction of a team's existing attack strength (λ)**,
  not their own free parameter.
- Penalty award rate varies ~3.7x (Real Madrid 0.26/match vs Burnley
  0.07/match) and is plausible as a real team effect (possession/attacking
  dominance draws more fouls in the box). This gets a real team-level model.
- Header goals cut across `situation` values, so must be identified via
  `shot_type == 'Head'`, not situation.

## Stage 1 — Dixon-Coles fit to pre-match odds

Rather than fitting team attack/defense parameters from historical goals,
we invert each match's own **pre-match 1X2 and Over/Under 2.5 odds**
directly into that match's implied (λ_home, λ_away, ρ). This sidesteps
needing a rolling historical team-strength model in Stage 1 entirely — the
market price already reflects everything relevant (form, injuries, etc.) as
of kickoff, so there's no "staleness" or decay question here the way there
is in Stage 2.

**The draw factor (ρ).** Plain independent Poisson(λ_home) x Poisson(λ_away)
underestimates how often football matches end in low-scoring draws.
Dixon-Coles corrects this with a multiplicative adjustment τ applied only to
the four low-score cells:

```
τ(0,0) = 1 - λ·μ·ρ
τ(1,0) = 1 + λ·ρ
τ(0,1) = 1 + μ·ρ
τ(1,1) = 1 - ρ
τ(x,y) = 1                       otherwise
```

`P(X=x, Y=y) = τ(x,y) · Poisson(x; λ) · Poisson(y; μ)`

**Solving per match.** After de-vigging the market odds (proportional
normalization: divide each implied probability by the sum of all implied
probabilities in that market), we have 3 independent constraints per match:
`P(Home)`, `P(Draw)` (equivalently `P(Away)`, since the three sum to 1), and
`P(Over 2.5)`. That's exactly enough to identify the 3 unknowns
`(λ_home, λ_away, ρ)` — solved numerically per match via least squares,
minimizing the gap between market-implied probabilities and the
Dixon-Coles-implied probabilities from a candidate `(λ, μ, ρ)`.

**Derived quantities:**

- `P(no_goal) = P(0,0)` directly from the fitted Dixon-Coles distribution.
- `P(home scores first) = λ_home / (λ_home + λ_away) · (1 - P(no_goal))`,
  and symmetrically for away. This is the standard result for a race
  between two independent Poisson-rate processes over the match — a
  simplification that ignores second-order dynamics (a team playing
  differently after conceding, red cards, etc.), noted as a modeling
  assumption to revisit if validation shows it's biased.

## Stage 2 — Method shares

For whichever team scores first, what's the breakdown across
`{shot, header, penalty, free_kick, own_goal}`?

We reuse λ from Stage 1 as the team's total goal rate for this fixture —
Stage 2 only needs to answer what *share* of that rate goes to each method,
which turns this into a compositional/multinomial problem rather than
requiring independent rate models for rare events.

- **shot** — not modeled directly. Defined as the residual:
  `1 - header_share - penalty_share - free_kick_share - own_goal_share`.
  It's ~70% of all goals; modeling the other four accurately and taking the
  complement is simpler and more robust than fitting a fifth category.
- **header** — team's headed-xG share of their total xG (attacking aerial
  threat), blended multiplicatively with the opponent's headed-xG-conceded
  share (aerial defensive weakness), in the same attack x defense spirit as
  Dixon-Coles itself.
- **penalty** — team's penalty-award rate x opponent's penalty-concession
  rate, both heavily shrunk toward league average (penalties are ~0.15 per
  team per match — noisy even over hundreds of matches).
- **free_kick** — league-average constant. Not team-specific (see findings
  above).
- **own_goal** — small fixed fraction of the team's own λ, not an
  independently fitted parameter.

### Recency weighting (header & penalty only)

Squad-level traits (aerial presence, a team's tendency to draw fouls) shift
on a transfer-window timescale, so older matches should count for less when
estimating these two rates. Stage 1 doesn't need this (see above — the odds
are already point-in-time), but Stage 2 aggregates a team's own history, so
it does.

- Exponential weighting: `weight = exp(-ln(2)/H * days_since_match)`,
  parameterized by half-life `H` (starting point: ~1.5-2 years, tuned via
  backtesting, not fixed by theory alone).
- Applied at the **goal-event level** for header share:
  `weighted_header_share = sum(w_i * is_header_i) / sum(w_i)`.
- Applied at the **match level** for penalty rate:
  `weighted_pen_rate = sum(w_m * pens_awarded_m) / sum(w_m)`.

### Avoiding overfitting on method shares

This is the main risk in Stage 2, addressed in layers:

1. **Shrinkage is not optional polish, it's the primary defense.** Every
   team rate is pulled toward a higher-level average, strength inversely
   proportional to *decay-weighted* effective sample size (`sum(w)`, not
   raw match count — decay already shrinks a team's effective history, so
   shrinkage has to account for that):
   `shrunk_rate = (sum(w)*x + k*prior) / (sum(w) + k)`.
2. **Hierarchical prior**: team -> league average -> global average, not
   straight to one global number, so small-sample leagues don't get either
   over-trusted or swamped by cross-league noise.
3. **Structural parsimony**: attack x defense multiplicative form (same as
   Dixon-Coles) instead of fitting every team-pair interaction
   independently, and `shot` as a residual instead of a 5th fitted category.
4. **Hyperparameters (`H`, `k`) tuned on held-out log-loss**, never chosen
   by eye on training data.
5. **Time-respecting validation only.** Chronological train/test split
   (e.g. train through 2022/23, validate 2023/24-2025/26). Never random
   k-fold — a random split could leak a team's future matches into their
   own decay-weighted historical rate for an earlier prediction.
6. **Baseline comparison is the actual bar to clear.** Every category's
   team-specific model must beat a naive "league-average constant share for
   everyone" baseline out-of-sample, or it gets shrunk further / dropped
   back to league-constant (this is what already happened to free-kicks and
   own-goals on inspection).
7. **Calibration checks**, not just aggregate log-loss — bucket predictions
   into deciles per category and compare actual vs. predicted frequency, to
   catch a model that's well-calibrated on average but overconfident for
   high-share teams specifically.

## Build order

1. `labels.py` — build the first-goal-of-match ground truth table from
   `shots` (this doc's classification rule + `no_goal` for 0-0s). Also
   serves as a sanity-check dataset: empirical no-goal rate should roughly
   match what Stage 1 predicts. **Done.** Verified against a real match
   with an own goal (SC Bastia 3-3 Marseille, match_id 6186) to confirm the
   scoring-side flip is correct. Distribution across 21,588 matches: shot
   64.4%, header 16.1%, penalty 7.7%, no_goal 6.9%, own_goal 3.0%,
   free_kick 2.0% — all in plausible ranges (no_goal in particular lines up
   with the ~8-9% typically quoted for top-flight football, on the low end
   but pooled across 5 leagues/12 seasons).
2. `odds.py` — de-vig market odds (1X2, O/U 2.5). **Done.**
3. `dixon_coles.py` — τ-adjusted bivariate Poisson + per-match solver for
   `(λ_home, λ_away, ρ)` from de-vigged odds, plus the derived
   no-goal/first-scorer quantities. **Done.** Stress-tested on 2,000 random
   matches: zero solver failures, max residual ~1e-7, ρ and implied total
   goals (mean 2.78/match) both land in expected ranges.

   **Open finding for the validation phase:** mean predicted P(no_goal)
   across the full dataset is 7.57% vs an empirical 6.85% (21,587 matches)
   — a real, if modest, systematic overestimate. Worth investigating during
   backtesting (candidates: the simple proportional de-vig not correcting
   for favorite-longshot bias, since that bias is usually most visible in
   extreme/rare outcomes like 0-0; or trying Shin's de-vig method instead).
   Not a solver bug — the fit itself is essentially exact against the
   (devigged) market odds it's given.
4. `method_shares.py` — decay weighting + shrinkage utilities, the
   header/penalty rate builders, and the free_kick/own_goal league-level
   constants. **Done.** Spot-checked qualitatively: Burnley's header share
   (0.183) notably exceeds Manchester City's (0.126), consistent with
   their contrasting styles; small-sample teams (Luton, 49 historical
   goals) get pulled substantially toward the league prior as intended.
5. Wire stages together, backtest against the label table (log-loss/Brier
   per category, holdout on 2024/25-2025/26), compare against the
   league-average baseline, and tune `half_life_days`/`pseudo_count` on
   held-out performance rather than the provisional defaults used above.

Steps 1-4 are implemented as a first pass in this folder. Step 5
(backtesting/validation harness) is the natural next piece of work.
