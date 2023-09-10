import matplotlib.pyplot as plt
from numba import njit
import math
import numpy as np


"""
Section 4 Agent-based Simulation of the article 
Epps Effect and the Signature of Short-term Momentum Traders
"""

# njit decorators used to speed up code (can be removed)


@njit
def round_tick(x, eta):
    return eta * math.floor(x / eta + 0.5)


@njit()
def run_simulation(params, h):
    # collect parameters

    T = int(params[0])
    np.random.seed(1)

    q = np.zeros(T * 2)
    q_n = np.zeros(T * 2)
    q_m = np.zeros(T * 2)
    s = np.zeros(T * 2)
    index = np.zeros(T)
    momentum_index = np.zeros(T * 2)
    rho = np.zeros((len(h), 3))

    s[0] = params[1]
    s[T] = params[2]
    eta = [params[3], params[4]]
    A = [params[5], params[6]]
    k = [params[7], params[8]]
    sigma = [params[9], params[10]]
    gamma = [params[11], params[12]]
    theta = [params[13], params[14]]
    psi_n = [params[15], params[16]]
    psi_m = [params[17], params[18]]
    q_m_max = params[19], params[20]
    tau = int(params[21])

    total_trades = [0.0, 0.0]
    delta_a = [0.0, 0.0]
    delta_b = [0.0, 0.0]

    for t in range(1, T):

        # compute moving average of window tau
        if t >= tau:
            momentum_index[t] = np.mean(index[t-tau:t])

        for i in range(2):
            total_trades[i] = 0.0
            s[t + i * T] = s[t - 1 + i * T]
            q_m[t + i * T] = q_m[t - 1 + i * T]
            q_n[t + i * T] = q_n[t - 1 + i * T]
            q[t + i * T] = q[t - 1 + i * T]

            # momentum traders
            if t >= tau + 1:
                if (s[t - 1] / s[0] + s[t + T - 1] / s[T]) > \
                        momentum_index[t]:
                    q_m[t + i * T] = min(max(q_m[t - 1 + i * T], 0.0) + psi_m[i], q_m_max[i])
                else:
                    q_m[t + i * T] = max(min(q_m[t - 1 + i * T], 0.0) - psi_m[i], -q_m_max[i])

            total_trades[i] += q_m[t + i * T] - q_m[t - 1 + i * T]

            # noise traders
            u = np.random.uniform(0, 1)
            dN_b = 0.0

            if u < A[i] * math.exp(-k[i] * delta_b[i]):
                dN_b = 1.0

            u = np.random.uniform(0, 1)
            dN_a = 0.0

            if u < A[i] * math.exp(-k[i] * delta_a[i]):
                dN_a = 1.0

            q_n[t + i * T] = q_n[t - 1 + i * T] + psi_n[i] * (dN_a - dN_b)
            total_trades[i] += psi_n[i] * (dN_a - dN_b)
            q[t + i * T] -= total_trades[i]

            # market makers
            delta_b[i] = round_tick(max(0.5 * gamma[i] * sigma[i] * sigma[i] * T + (1. / gamma[i])
                       * math.log(1. + gamma[i] / k[i]) + gamma[i] * sigma[i] * sigma[i] * T
                       * q[t + i * T], 0.), eta[i])
            delta_a[i] = round_tick(max(0.5 * gamma[i] * sigma[i] * sigma[i] * T + (1. / gamma[i])
                       * math.log(1. + gamma[i] / k[i]) - gamma[i] * sigma[i] * sigma[i] * T
                       * q[t + i * T], 0.), eta[i])

            dZ = np.random.normal(0, 1)
            s[t + i * T] = max(round_tick(s[t - 1 + i * T] + theta[i] * total_trades[i] + sigma[i] * dZ, eta[i]), eta[i])
        index[t] = s[t] / s[0] + s[t + T] / s[T]

    for i in range(len(h)):
        tenor = h[i]
        rho[i] = compute_correlation(s, tenor, T)

    return rho, s


@njit(parallel=True, fastmath=False)
def compute_correlation(s: np.ndarray, tenor: int, T: int):

    rho = np.zeros(3)
    if tenor < T / 4.:
        n_effective = int(T / tenor)

        log_returns = np.zeros((T - tenor, 2), dtype=np.float64)

        for i in range(2):
            s_i = s[i*T:T + i*T]
            log_returns[:, i] = np.log(s_i[tenor:]) - np.log(s_i[:-tenor])

        rho[1] = np.corrcoef(log_returns[:, 0], log_returns[:, 1])[0, 1]
        rho[0] = np.tanh(np.arctanh(rho[1]) - 2. / np.sqrt(n_effective - 3))
        rho[2] = np.tanh(np.arctanh(rho[1]) + 2. / np.sqrt(n_effective - 3))

    return rho


scale_factor = 2  # dt = 0.5s

# simulation parameters
T = 20_000_000
nassets = 2
A = np.array([1, 1])
s0 = np.array([1.10, 30000.])
eta = [1e-4, 1e-2]
psi_m = [3000000., 120.]
psi_n = [100000., 4.]
q_m_max = [6500000., 250.]
k = np.array([3466, 34.66])
theta = np.array([2.7e-11, 2.7e-6])
gamma = np.array([6.46e-6, 3.47e-9])
sigma = np.array([3e-5, 2.04])
tau = 1000

h = np.array(np.arange(1, 501))

params = np.array([T, s0[0], s0[1], eta[0], eta[1], A[0], A[1],
                   k[0], k[1], sigma[0], sigma[1], gamma[0], gamma[1],
                   theta[0], theta[1], psi_n[0], psi_n[1], psi_m[0], psi_m[1],
                   q_m_max[0], q_m_max[1], tau])

# run simulation
rho, s = run_simulation(params, h)

plt.figure()
plt.plot(s[:T])
plt.show()
plt.figure()
plt.plot(s[T:])
plt.show()

# display correlation
ind = int(len(rho) / scale_factor)
plt.figure()
plt.plot(h[::scale_factor], rho[:ind, 1])
plt.plot(h[::scale_factor], rho[:ind, 0], c='r', linestyle="dotted")
plt.plot(h[::scale_factor], rho[:ind, 2], c='r', linestyle="dotted")
plt.title(r'$\rho(h)$ with Fisher confidence interval')
plt.xlabel(r'h (seconds)')
plt.show()
