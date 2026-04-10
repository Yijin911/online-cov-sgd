"""
Numerical experiments for online covariance estimation in averaged SGD.

Experiment 1: Rate comparison with growing blocks
  - Quadratic objective F(x) = (1/2) x^T H x, S = I, V = H^{-2}
  - Growing blocks a_m = floor(C * m^beta)
  - Compare: original BM (rho=1, beta*), improved BM (beta_dagger),
    burn-in BM (rho=0.5, beta_dagger), trajectory regression

Experiment 2: Per-block bias-variance decomposition (growing blocks)
  - For each block index m, compute empirical bias and variance
  - Show that early blocks dominate bias (motivating burn-in)

Usage:
    python run_experiments.py
"""
import numpy as np
from numpy.linalg import norm as la_norm
import numba as nb
import json
import os
import time


def build_block_schedule(n, beta, C=5.0):
    """Pre-compute growing block boundaries."""
    blocks = []
    t, m = 0, 1
    while t < n:
        a_m = max(1, int(C * m ** beta))
        if t + a_m > n:
            break
        blocks.append((t, t + a_m))
        t += a_m
        m += 1
    return blocks


@nb.njit
def sgd_online_all(n, d, alpha, eta0, eigenvalues,
                   starts_1, ends_1, starts_2, ends_2, seed):
    """Run SGD once, accumulate block sums for two schedules AND
    regression sufficient statistics, all in a single pass (O(d^2) memory)."""
    np.random.seed(seed)
    nb1 = len(starts_1)
    nb2 = len(starts_2)

    z = np.zeros(d)
    z[:] = np.random.randn(d)
    global_sum = np.zeros(d)

    # Block accumulators for schedule 1
    bs1 = np.zeros((nb1, d))
    bsz1 = np.zeros(nb1, dtype=np.int64)
    cur1 = np.zeros(d)
    bi1 = 0

    # Block accumulators for schedule 2
    bs2 = np.zeros((nb2, d))
    bsz2 = np.zeros(nb2, dtype=np.int64)
    cur2 = np.zeros(d)
    bi2 = 0

    # Regression sufficient statistics (in eigenbasis)
    # y_t = H_eig * x_t + zeta_t, regress y on x per coordinate
    sum_x = np.zeros(d)       # sum of x_t
    sum_y = np.zeros(d)       # sum of y_t
    sum_xx = np.zeros(d)      # sum of x_t^2
    sum_xy = np.zeros(d)      # sum of x_t * y_t
    sum_yy = np.zeros(d)      # sum of y_t^2
    count = 0

    for t in range(n):
        eta_t = eta0 * (t + 1) ** (-alpha)
        eps = np.random.randn(d)
        z_new = np.zeros(d)
        for i in range(d):
            z_new[i] = (1.0 - eta_t * eigenvalues[i]) * z[i] + eta_t * eps[i]

        # y_t = (z - z_new) / eta_t  (in eigenbasis)
        for i in range(d):
            y_i = (z[i] - z_new[i]) / eta_t
            sum_x[i] += z[i]
            sum_y[i] += y_i
            sum_xx[i] += z[i] * z[i]
            sum_xy[i] += z[i] * y_i
            sum_yy[i] += y_i * y_i
        count += 1

        z[:] = z_new
        global_sum += z

        # Block schedule 1
        if bi1 < nb1 and t >= starts_1[bi1]:
            cur1 += z
            if t + 1 == ends_1[bi1]:
                bs1[bi1] = cur1.copy()
                bsz1[bi1] = ends_1[bi1] - starts_1[bi1]
                cur1[:] = 0.0
                bi1 += 1

        # Block schedule 2
        if bi2 < nb2 and t >= starts_2[bi2]:
            cur2 += z
            if t + 1 == ends_2[bi2]:
                bs2[bi2] = cur2.copy()
                bsz2[bi2] = ends_2[bi2] - starts_2[bi2]
                cur2[:] = 0.0
                bi2 += 1

    return (global_sum, bs1, bsz1, bs2, bsz2,
            sum_x, sum_y, sum_xx, sum_xy, sum_yy, count)


def compute_bm_error(block_sums_eig, block_sizes, global_sum_eig, n, Q,
                     V_true, rho):
    """Compute growing-block batch-means estimator error."""
    d = Q.shape[0]
    b_n = int(np.sum(block_sizes > 0))
    if b_n == 0:
        return la_norm(V_true, ord=2)

    xbar = Q @ (global_sum_eig / n)
    K_n = max(1, int(rho * b_n))
    start_idx = b_n - K_n

    Sigma_hat = np.zeros((d, d))
    for i in range(start_idx, b_n):
        a_m = block_sizes[i]
        Y_m = (Q @ block_sums_eig[i] - a_m * xbar) / np.sqrt(a_m)
        Sigma_hat += np.outer(Y_m, Y_m)
    Sigma_hat /= K_n

    return la_norm(Sigma_hat - V_true, ord=2)


def compute_regression_error(sum_x, sum_y, sum_xx, sum_xy, sum_yy,
                             count, Q, eigenvalues_H, V_true):
    """Trajectory-regression estimator error from sufficient statistics."""
    d = Q.shape[0]
    n = count

    # Centred regression per coordinate (eigenbasis)
    x_bar = sum_x / n
    y_bar = sum_y / n
    H_hat_diag = np.zeros(d)
    S_hat_diag = np.zeros(d)  # residual variance per coordinate
    for i in range(d):
        sxx = sum_xx[i] - n * x_bar[i] ** 2
        sxy = sum_xy[i] - n * x_bar[i] * y_bar[i]
        if sxx > 1e-12:
            H_hat_diag[i] = sxy / sxx
        else:
            H_hat_diag[i] = eigenvalues_H[i]
        # Residual variance: Var(y - h*x) = Syy - h^2 * Sxx
        syy = sum_yy[i] - n * y_bar[i] ** 2
        S_hat_diag[i] = max(0.0, (syy - H_hat_diag[i] ** 2 * sxx) / n)

    # S_hat in eigenbasis is approximately diag (since noise is isotropic)
    # For V_hat = H_hat^{-1} S_hat H_hat^{-1}, work in original basis
    H_hat = Q @ np.diag(H_hat_diag) @ Q.T
    S_hat = Q @ np.diag(S_hat_diag) @ Q.T
    try:
        H_hat_inv = np.linalg.inv(H_hat)
    except np.linalg.LinAlgError:
        return la_norm(V_true, ord=2)

    V_hat = H_hat_inv @ S_hat @ H_hat_inv
    return la_norm(V_hat - V_true, ord=2)


def run_experiment1(output_dir):
    """Rate comparison: original BM, improved BM, burn-in BM, regression."""
    d = 10
    alpha_values = [0.55, 0.6, 0.7]
    n_values = [1000, 3000, 10000, 30000, 100000, 300000, 1000000,
                3000000, 10000000]
    eta0 = 1.0

    def get_n_reps(n):
        if n <= 100000:
            return 200
        elif n <= 1000000:
            return 100
        else:
            return 50

    rng = np.random.default_rng(42)
    eigenvalues_H = np.linspace(1.0, 5.0, d)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    H_inv = np.linalg.inv(Q @ np.diag(eigenvalues_H) @ Q.T)
    V_true = H_inv @ H_inv

    # Warmup numba
    print("  Warming up JIT...", flush=True)
    _dummy = sgd_online_all(100, d, 0.55, eta0, eigenvalues_H,
                            np.array([0], dtype=np.int64),
                            np.array([50], dtype=np.int64),
                            np.array([0], dtype=np.int64),
                            np.array([50], dtype=np.int64), seed=0)
    print("  JIT ready.", flush=True)

    results = {}
    for alpha in alpha_values:
        beta_original = 2.0 / (1.0 - alpha)
        beta_dagger = (1.0 + 2.0 * alpha) / (2.0 * (1.0 - alpha))

        for n in n_values:
            n_reps = get_n_reps(n)

            sched_orig = build_block_schedule(n, beta_original)
            sched_dag = build_block_schedule(n, beta_dagger)

            s1 = np.array([s for s, e in sched_orig], dtype=np.int64)
            e1 = np.array([e for s, e in sched_orig], dtype=np.int64)
            s2 = np.array([s for s, e in sched_dag], dtype=np.int64)
            e2 = np.array([e for s, e in sched_dag], dtype=np.int64)

            errors_orig = []
            errors_dag = []
            errors_dag_bi = []
            errors_reg = []

            t0 = time.time()
            for rep in range(n_reps):
                (gs, bs1, bsz1, bs2, bsz2,
                 sx, sy, sxx, sxy, syy, cnt) = sgd_online_all(
                    n, d, alpha, eta0, eigenvalues_H,
                    s1, e1, s2, e2, seed=rep)

                err_orig = compute_bm_error(
                    bs1, bsz1, gs, n, Q, V_true, rho=1.0)
                err_dag = compute_bm_error(
                    bs2, bsz2, gs, n, Q, V_true, rho=1.0)
                err_dag_bi = compute_bm_error(
                    bs2, bsz2, gs, n, Q, V_true, rho=0.5)
                err_reg = compute_regression_error(
                    sx, sy, sxx, sxy, syy, cnt, Q, eigenvalues_H, V_true)

                errors_orig.append(err_orig)
                errors_dag.append(err_dag)
                errors_dag_bi.append(err_dag_bi)
                errors_reg.append(err_reg)

            elapsed = time.time() - t0
            key = f"alpha={alpha},n={n}"
            results[key] = {
                "alpha": alpha,
                "n": n,
                "n_reps": n_reps,
                "original_mean": float(np.mean(errors_orig)),
                "original_std": float(np.std(errors_orig)),
                "improved_mean": float(np.mean(errors_dag)),
                "improved_std": float(np.std(errors_dag)),
                "burnin_mean": float(np.mean(errors_dag_bi)),
                "burnin_std": float(np.std(errors_dag_bi)),
                "regression_mean": float(np.mean(errors_reg)),
                "regression_std": float(np.std(errors_reg)),
            }
            print(f"  {key} ({n_reps} reps, {elapsed:.1f}s): "
                  f"orig={np.mean(errors_orig):.4f}, "
                  f"impr={np.mean(errors_dag):.4f}, "
                  f"bi={np.mean(errors_dag_bi):.4f}, "
                  f"reg={np.mean(errors_reg):.4f}", flush=True)

    with open(os.path.join(output_dir, "experiment1.json"), "w") as f:
        json.dump(results, f, indent=2)


def run_experiment2_bias_variance(output_dir):
    """Per-block bias and variance decomposition (growing blocks)."""
    d = 10
    alpha = 0.55
    n = 100000
    beta = (1.0 + 2.0 * alpha) / (2.0 * (1.0 - alpha))
    n_reps = 500
    eta0 = 1.0

    rng = np.random.default_rng(42)
    eigenvalues_H = np.linspace(1.0, 5.0, d)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    H_inv = np.linalg.inv(Q @ np.diag(eigenvalues_H) @ Q.T)
    V_true = H_inv @ H_inv

    sched = build_block_schedule(n, beta)
    n_blocks = len(sched)

    starts = np.array([s for s, e in sched], dtype=np.int64)
    ends = np.array([e for s, e in sched], dtype=np.int64)
    empty_s = np.zeros(0, dtype=np.int64)
    empty_e = np.zeros(0, dtype=np.int64)

    all_block_outers = [[None] * n_reps for _ in range(n_blocks)]

    for rep in range(n_reps):
        if rep % 100 == 0:
            print(f"  Bias-variance rep {rep}/{n_reps}", flush=True)

        (gs, bs, bsz, _, _,
         _, _, _, _, _, _) = sgd_online_all(
            n, d, alpha, eta0, eigenvalues_H,
            starts, ends, empty_s, empty_e, seed=rep)
        xbar = Q @ (gs / n)

        for m_idx in range(len(bsz)):
            if bsz[m_idx] == 0:
                break
            a_m = bsz[m_idx]
            Y_m = (Q @ bs[m_idx] - a_m * xbar) / np.sqrt(a_m)
            all_block_outers[m_idx][rep] = np.outer(Y_m, Y_m)

    block_stats = []
    for m_idx in range(n_blocks):
        outers = [o for o in all_block_outers[m_idx] if o is not None]
        if len(outers) == 0:
            continue
        outers = np.array(outers)
        mean_outer = outers.mean(axis=0)
        bias = la_norm(mean_outer - V_true, ord=2)
        variance = np.mean([la_norm(o - mean_outer, ord=2) for o in outers])
        block_stats.append({
            "block": int(m_idx),
            "bias": float(bias),
            "variance": float(variance),
        })

    with open(os.path.join(output_dir, "experiment2.json"), "w") as f:
        json.dump(block_stats, f, indent=2)


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(output_dir, exist_ok=True)

    print("Experiment 1: Rate comparison (4 estimators)", flush=True)
    run_experiment1(output_dir)

    print("\nExperiment 2: Per-block bias-variance decomposition", flush=True)
    run_experiment2_bias_variance(output_dir)

    print("\nDone. Results saved to figures/", flush=True)
