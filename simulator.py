import numpy as np
import math

# ------------- Poisson helpers ----------------

def poisson_pmf(k, lam):
    """Probability mass function of Poisson(lam) for k goals."""
    return math.exp(-lam) * lam**k / math.factorial(k)

def sample_poisson(lam, max_goals=10):
    """Randomly sample a Poisson(lam) for goals up to max_goals."""
    # Using numpy for random Poisson sampling:
    return np.random.poisson(lam)

# ---------- Dixon-Coles joint probability ---------

def dixon_coles_joint(x, y, lam1, lam2, rho):
    """
    Modified joint probability P(X=x, Y=y) using the Dixon-Coles correction factor
    for low goals (0 or 1 goals for each team).
    """
    # basic independent probability
    p = poisson_pmf(x, lam1) * poisson_pmf(y, lam2)
    # correction factor tau
    if x == 0 and y == 0:
        tau = 1 - lam1 * lam2 * rho
    elif x == 0 and y == 1:
        tau = 1 + lam1 * rho
    elif x == 1 and y == 0:
        tau = 1 + lam2 * rho
    elif x == 1 and y == 1:
        tau = 1 - rho
    else:
        tau = 1
    return p * tau

def sample_dixon_coles(lam1, lam2, rho, max_goals=5):
    """
    Simulate a random (goals_home, goals_away) sample from the Dixon-Coles distribution.
    We normalize the joint mass over the limited grid {0..max_goals} × {0..max_goals}
    to get sampling probabilities.
    """
    # build probability grid
    probs = []
    grid = []
    for x in range(max_goals+1):
        for y in range(max_goals+1):
            prob = dixon_coles_joint(x, y, lam1, lam2, rho)
            probs.append(prob)
            grid.append((x, y))

    # normalize
    total = sum(probs)
    probs = [p/total for p in probs]

    # sample index
    idx = np.random.choice(len(grid), p=probs)
    return grid[idx]

# --------------- Simulation functions ----------------

def simulate_match_poisson(home_team, away_team, params):
    """
    Simulate match with independent Poisson scoring.
    params should contain attack, defense, home_adv for each team.
    """
    lam_home = params['attack'][home_team] * params['defense'][away_team] * params['home_adv']
    lam_away = params['attack'][away_team] * params['defense'][home_team]
    goals_home = sample_poisson(lam_home)
    goals_away = sample_poisson(lam_away)
    return goals_home, goals_away

def simulate_match_dixon_coles(home_team, away_team, params, rho=0.0):
    """
    Simulate using the Dixon-Coles joint distribution with dependence parameter rho.
    """
    lam_home = params['attack'][home_team] * params['defense'][away_team] * params['home_adv']
    lam_away = params['attack'][away_team] * params['defense'][home_team]

    return sample_dixon_coles(lam_home, lam_away, rho)

# ---------------- Example usage ------------------

if __name__ == "__main__":
    # simple team strength parameters
    team_params = {
        'attack': {'TeamA': 1.3, 'TeamB': 1.1},
        'defense': {'TeamA': 0.9, 'TeamB': 1.0},
        'home_adv': 1.15  # average home advantage
    }

    # simulate 10 matches
    print("Poisson simulation (independent):")
    for i in range(10):
        h, a = simulate_match_poisson("TeamA", "TeamB", team_params)
        print(f"Match {i+1}: TeamA {h} - {a} TeamB")

    print("\nDixon-Coles simulation (with rho correction):")
    for i in range(10):
        h, a = simulate_match_dixon_coles("TeamA", "TeamB", team_params, rho=0.05)
        print(f"Match {i+1}: TeamA {h} - {a} TeamB")
