# Online Covariance Estimation in Averaged SGD

Simulation code for the paper:

> **Online Covariance Estimation in Averaged SGD: Improved Batch-Means Rates and Minimax Optimality via Trajectory Regression**
> Yijin Ni and Xiaoming Huo

## Requirements

- Python 3.8+
- NumPy
- Numba
- Matplotlib

Install dependencies:
```bash
pip install numpy numba matplotlib
```

## Usage

**Run experiments:**
```bash
python run_experiments.py
```
This produces JSON result files for both experiments:
1. **Rate comparison** — compares operator-norm convergence of original batch means, improved batch means (optimally tuned block growth), burn-in batch means, and trajectory regression.
2. **Per-block bias-variance decomposition** — shows that early blocks dominate bias, motivating the burn-in modification.

**Generate figures:**
```bash
python plot_figures.py
```
This reads the JSON results and produces:
- `figures/rate_comparison.pdf` (Figure 1 in the paper)
- `figures/bias_variance_decomp.pdf` (Figure 2 in the paper)

## Experimental Setup

- Quadratic objective: F(x) = (1/2) x'Hx with isotropic noise S = I, so V = H^{-2}
- Growing block schedule: a_m = floor(C * m^beta)
- Learning rate: eta_t = eta_0 * t^{-alpha}, alpha in (1/2, 1)

## License

MIT
