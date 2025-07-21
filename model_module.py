import numpy as np
from scipy.integrate import solve_ivp

def ode_rhs(t, a, log_k1, log_k2, m, n, r, eps=1e-10):
    k1 = 10 ** log_k1
    k2 = 10 ** log_k2
    a = np.clip(a, eps, r - eps)
    return (k1 + k2 * a**m) * (1 - a)**(n/2) * (r - a)**(n/2)

def solve_model(log_k1, log_k2, m, n, r, t_data, a_data, eps=1e-10):
    try:
        k1 = 10 ** log_k1
        k2 = 10 ** log_k2
        a0 = np.clip(a_data[0], eps, r - eps)

        sol = solve_ivp(ode_rhs, [t_data[0], t_data[-1]], [a0], t_eval=t_data,
                        args=(log_k1, log_k2, m, n, r), method='LSODA',
                        rtol=1e-6, atol=1e-8)

        if not sol.success or not np.all(np.isfinite(sol.y)):
            return np.full_like(t_data, np.nan)

        return sol.y[0]

    except Exception as e:
        print(f"ODE solver failed with exception: {e}")
        return np.full_like(t_data, np.nan)
  
def log_prior(params, t_data, a_data):
    log_k1, log_k2, m, n, r, log_sigma = params
    if not (-8 < log_k1 < -1 and
            -8 < log_k2 < -1 and
             0 < m < 3 and
             0 < n < 3 and
             0.99 * np.max(a_data) < r < 1.01 * np.max(a_data) and
            -10 < log_sigma < 0):
        return -np.inf
    return 0.0

def log_likelihood(params, t_data, a_data):
    log_k1, log_k2, m, n, r, log_sigma = params
    a_fit = solve_model(log_k1, log_k2, m, n, r, t_data, a_data)

    if np.any(np.isnan(a_fit)) or np.any(a_fit <= 0) or np.any(a_fit >= r):
        return 1e6  # use large penalty for optimizer compatibility

    sigma = 10 ** log_sigma
    residual = (a_data - a_fit) / sigma
    return -0.5 * np.sum(residual**2 + np.log(2 * np.pi * sigma**2))

def log_posterior(params, t_data, a_data):
    lp = log_prior(params, t_data, a_data)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, t_data, a_data)