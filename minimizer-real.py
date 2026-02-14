import numpy as np
import math
from datetime import datetime
from scipy.optimize import minimize

#############################################
# Dixon–Coles low score correction
#############################################

def tau_correction(x, y, lam, mu, rho):
    if x == 0 and y == 0:
        return 1 - lam * mu * rho
    elif x == 0 and y == 1:
        return 1 + lam * rho
    elif x == 1 and y == 0:
        return 1 + mu * rho
    elif x == 1 and y == 1:
        return 1 - rho
    else:
        return 1.0


#############################################
# Build dataset from your JSON structure
#############################################

def prepare_data(json_data):
    matches = json_data["matches"]

    teams = sorted(
        list(set(m["team1"] for m in matches) |
             set(m["team2"] for m in matches))
    )
    team_index = {team: i for i, team in enumerate(teams)}

    # Parse dates
    dates = [datetime.strptime(m["date"], "%Y-%m-%d") for m in matches]
    most_recent = max(dates)

    data = []

    for m in matches:
        home = team_index[m["team1"]]
        away = team_index[m["team2"]]

        x = m["score"]["ft"][0]
        y = m["score"]["ft"][1]

        match_date = datetime.strptime(m["date"], "%Y-%m-%d")
        days_diff = (most_recent - match_date).days

        data.append((home, away, x, y, days_diff))

    return data, teams


#############################################
# Negative Log Likelihood with time decay
#############################################

def dc_neg_log_likelihood(params, data, n_teams, xi):
    """
    params structure:
    [ attack_0,...attack_n-1,
      defense_0,...defense_n-1,
      home_adv,
      rho ]
    """

    attacks = params[0:n_teams]
    defenses = params[n_teams:2*n_teams]
    home_adv = params[-2]
    rho = params[-1]

    log_likelihood = 0.0

    for home, away, x, y, days_diff in data:

        lam = np.exp(attacks[home] - defenses[away] + home_adv)
        mu = np.exp(attacks[away] - defenses[home])

        # Time decay weight
        weight = np.exp(-xi * days_diff)

        # Poisson components
        log_p = (
            x * np.log(lam) - lam - math.lgamma(x + 1)
            + y * np.log(mu) - mu - math.lgamma(y + 1)
        )

        # Dixon-Coles correction
        tau = tau_correction(x, y, lam, mu, rho)

        if tau <= 0:
            return 1e10  # penalize invalid regions

        log_likelihood += weight * (log_p + np.log(tau))

    return -log_likelihood  # minimize


#############################################
# Fit function
#############################################

def fit_dixon_coles(json_data, xi=0.001):

    data, teams = prepare_data(json_data)
    n_teams = len(teams)

    # Initial parameters
    init_attacks = np.zeros(n_teams)
    init_defenses = np.zeros(n_teams)
    init_home = 0.1
    init_rho = 0.0

    initial_params = np.concatenate([
        init_attacks,
        init_defenses,
        [init_home],
        [init_rho]
    ])

    # Constraint: sum(attacks) = 0 (identifiability)
    def constraint(params):
        return np.sum(params[0:n_teams])

    cons = {'type': 'eq', 'fun': constraint}

    result = minimize(
        dc_neg_log_likelihood,
        initial_params,
        args=(data, n_teams, xi),
        constraints=cons,
        method='SLSQP'
    )

    return result, teams


#############################################
# Example usage
#############################################

if __name__ == "__main__":

    import json

    with open("prem_data.json") as f:
        league_data = json.load(f)

    result, teams = fit_dixon_coles(league_data, xi=0.002)

    print("Optimization success:", result.success)
    print("Home advantage:", result.x[-2])
    print("Rho:", result.x[-1])

    print("\nAttack strengths:")
    for i, team in enumerate(teams):
        print(team, result.x[i])

    print("\nDefense strengths:")
    for i, team in enumerate(teams):
        print(team, result.x[i + len(teams)])
