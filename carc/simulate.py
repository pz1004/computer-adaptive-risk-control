"""
Synthetic early-exit network simulator.

Generative model (per input):
  - difficulty d in [0, 1];  source: d ~ Uniform[0,1];  target: d ~ exp-tilt p(d) prop. exp(lam*d)
  - at exit l (l = 0..L-1), competence q_l(d) = sigmoid(beta_l - gamma*d), beta increasing in l
    => deeper exits are more competent (Assumption M holds for the global-threshold chain)
  - correctness  c_l ~ Bernoulli(q_l(d))
  - confidence   s_l ~ Beta(5,2) if correct else Beta(2,5)   (informative but noisy)
Exit policy with global threshold t: exit at first l with s_l >= t, else exit L-1.
Loss = 1 - correctness at the exit reached;  cost = cost_of(exit reached).

The chain of configurations is the grid of global thresholds `self.thresholds`
(index 0 = lowest threshold = cheapest;  index K-1 = highest threshold = most expensive).
This matches selector.py's "increasing compute with index" convention.

Covariate shift acts only on the X-distribution (difficulty d); Y|X is unchanged, so the
exact importance weight w(d) = p_target(d)/p_source(d) = exp(lam*d) * lam/(exp(lam)-1).
"""
from __future__ import annotations
import numpy as np
from .chain import build_chain


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class EarlyExitSim:
    def __init__(self, L=5, gamma=4.0, beta0=-1.0, beta_step=1.6, K=40,
                 t_lo=0.40, t_hi=0.98, miscalib_exit=None, seed=0):
        self.L = L
        self.gamma = gamma
        self.betas = beta0 + beta_step * np.arange(L)        # increasing competence with depth
        self.costs_exit = np.arange(1, L + 1, dtype=float)    # cumulative cost per exit
        self.thresholds = np.linspace(t_lo, t_hi, K)          # the compute chain (asc. cost)
        self.K = K
        self.miscalib_exit = miscalib_exit                    # if set, that exit is overconfident-but-wrong
        self.rng_master = np.random.default_rng(seed)

    # ---- low-level sampling of latent (difficulty, correctness, confidence) ----
    def _draw_latents(self, n, rng, lam=0.0):
        if lam == 0.0:
            d = rng.uniform(0.0, 1.0, size=n)
        else:
            # inverse-CDF of exp-tilt density on [0,1]:  F(d) = (e^{lam d}-1)/(e^{lam}-1)
            u = rng.uniform(0.0, 1.0, size=n)
            d = np.log1p(u * (np.exp(lam) - 1.0)) / lam
        q = _sigmoid(self.betas[None, :] - self.gamma * d[:, None])   # (n, L)
        correct = (rng.uniform(size=q.shape) < q).astype(float)
        # confidence: high if correct, low if wrong
        conf = np.where(correct > 0.5,
                        rng.beta(5.0, 2.0, size=q.shape),
                        rng.beta(2.0, 5.0, size=q.shape))
        if self.miscalib_exit is not None:
            e = self.miscalib_exit
            # this exit is overconfident even when wrong -> breaks monotonicity of risk
            conf[:, e] = np.where(correct[:, e] > 0.5,
                                  rng.beta(5.0, 2.0, size=n),
                                  rng.beta(6.0, 2.0, size=n))
        return d, correct, conf

    def sample(self, n, rng=None, lam=0.0):
        """Return loss_matrix (n,K), cost_matrix (n,K), and difficulty d (n,)."""
        if rng is None:
            rng = self.rng_master
        d, correct, conf = self._draw_latents(n, rng, lam=lam)
        loss, cost = build_chain(conf, 1.0 - correct, self.costs_exit, self.thresholds)
        return loss, cost, d

    def oracle(self, big_n=400_000, lam=0.0, seed=12345):
        """Ground-truth true risk R_k and true cost C_k per config (large Monte Carlo)."""
        rng = np.random.default_rng(seed)
        loss, cost, _ = self.sample(big_n, rng=rng, lam=lam)
        return loss.mean(axis=0), cost.mean(axis=0)

    # exact importance weight for the exp-tilt shift (used as oracle in shift experiments)
    @staticmethod
    def exact_weight(d, lam):
        if lam == 0.0:
            return np.ones_like(d)
        Z = (np.exp(lam) - 1.0) / lam
        return np.exp(lam * d) / Z


def fit_weights_logistic(d_src, d_tgt, n_iter=300, lr=0.5):
    """
    Estimate the density ratio w(d) = p_tgt/p_src by probabilistic classification
    (src=0, tgt=1) with features [1, d, d^2], then w_hat = (p/(1-p)) * (n_src/n_tgt).
    Returns a callable d -> w_hat(d). Plain numpy logistic regression (no deps).
    """
    d_src = np.asarray(d_src, float); d_tgt = np.asarray(d_tgt, float)
    X = np.concatenate([d_src, d_tgt])
    y = np.concatenate([np.zeros(len(d_src)), np.ones(len(d_tgt))])
    Phi = np.stack([np.ones_like(X), X, X * X], axis=1)
    w = np.zeros(3)
    m = len(y)
    for _ in range(n_iter):
        p = _sigmoid(Phi @ w)
        grad = Phi.T @ (p - y) / m
        w -= lr * grad
    ratio_const = len(d_src) / max(len(d_tgt), 1)

    def wfun(d):
        d = np.asarray(d, float)
        phi = np.stack([np.ones_like(d), d, d * d], axis=1)
        p = _sigmoid(phi @ w)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return (p / (1 - p)) * ratio_const
    return wfun
