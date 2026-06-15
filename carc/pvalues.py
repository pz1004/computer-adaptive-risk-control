"""
Valid p-values for the null  H_tau: R(tau) > alpha  (reject => certify tau as safe).

All p-values are super-uniform under H_tau (i.e. P_{R>=alpha}(p <= u) <= u), which is
exactly the property the Learn-then-Test FWER machinery in `certify.py` requires.

Each function returns a p-value in (0, 1]. A *small* p-value is evidence that R(tau) <= alpha.

References
----------
- Hoeffding (1963); the simple form is Eq. (2) of the draft.
- Hoeffding-Bentkus (HB): Bates, Angelopoulos, Lei, Malik, Jordan (2021); Angelopoulos et al., LTT.
- Empirical Bernstein: Maurer & Pontil (2009), one-sided, inverted to a p-value.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import binom

__all__ = [
    "hoeffding_simple", "hoeffding_kl", "hoeffding_bentkus",
    "empirical_bernstein", "weighted_hoeffding", "weighted_empirical_bernstein",
]


# ---------------------------------------------------------------------------
# Unweighted p-values for losses in [0, 1]
# ---------------------------------------------------------------------------
def hoeffding_simple(rhat: float, n: int, alpha: float) -> float:
    """Eq. (2): exp(-2 n (alpha - rhat)_+^2).  Valid, but the loosest of the three."""
    d = max(alpha - rhat, 0.0)
    return float(np.exp(-2.0 * n * d * d))


def _binary_kl(a: float, b: float) -> float:
    """KL( Bernoulli(a) || Bernoulli(b) ), clipped for numerical safety."""
    eps = 1e-12
    a = min(max(a, eps), 1 - eps)
    b = min(max(b, eps), 1 - eps)
    return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))


def hoeffding_kl(rhat: float, n: int, alpha: float) -> float:
    """Relative-entropy (Chernoff) form: exp(-n * KL(rhat, alpha)). Tighter than the simple form."""
    if rhat >= alpha:
        return 1.0
    return float(min(np.exp(-n * _binary_kl(rhat, alpha)), 1.0))


def hoeffding_bentkus(rhat: float, n: int, alpha: float) -> float:
    """HB p-value = min( Hoeffding-KL term , e * Bentkus binomial term ). The LTT default."""
    if rhat >= alpha:
        return 1.0
    g1 = np.exp(-n * _binary_kl(rhat, alpha))               # Hoeffding (KL) term
    g2 = np.e * binom.cdf(np.ceil(n * rhat), n, alpha)       # Bentkus term
    return float(min(min(g1, g2), 1.0))


def empirical_bernstein(losses: np.ndarray, alpha: float, rng: float = 1.0) -> float:
    """
    Empirical-Bernstein p-value for losses in [0, rng], obtained by inverting the
    one-sided Maurer-Pontil bound

        P( R - Rhat >= sqrt(2 V ln(1/d)/n) + 7 rng ln(1/d)/(3(n-1)) ) <= d .

    With u = ln(1/d), solving alpha - Rhat = sqrt(2V/n) sqrt(u) + (7 rng/3(n-1)) u  for u
    gives p = exp(-u).  Variance V is the (unbiased) sample variance of `losses`.
    """
    losses = np.asarray(losses, dtype=float)
    n = losses.size
    rhat = losses.mean()
    if rhat >= alpha:
        return 1.0
    V = losses.var(ddof=1) if n > 1 else 0.0
    c = alpha - rhat
    a = np.sqrt(2.0 * V / n)                       # coefficient on sqrt(u)
    b = (7.0 * rng) / (3.0 * (n - 1)) if n > 1 else (7.0 * rng) / 3.0   # coefficient on u
    if b <= 0:
        u = (c / a) ** 2 if a > 0 else np.inf
    else:
        s = (-a + np.sqrt(a * a + 4.0 * b * c)) / (2.0 * b)   # s = sqrt(u) >= 0
        u = s * s
    return float(min(np.exp(-u), 1.0))


# ---------------------------------------------------------------------------
# Weighted p-values for covariate shift: weighted losses in [0, W] (Theorem 4).
# `wlosses` are the products w_hat_i * loss_i; `alpha_def` is the deflated target alpha - eta.
# ---------------------------------------------------------------------------
def weighted_hoeffding(wlosses: np.ndarray, alpha_def: float, W: float) -> float:
    """Range-W Hoeffding p-value against the deflated target alpha_def = alpha - eta."""
    wlosses = np.asarray(wlosses, dtype=float)
    n = wlosses.size
    rqhat = wlosses.mean()
    d = max(alpha_def - rqhat, 0.0)
    return float(np.exp(-2.0 * n * d * d / (W * W)))


def weighted_empirical_bernstein(wlosses: np.ndarray, alpha_def: float, W: float) -> float:
    """Empirical-Bernstein p-value for weighted losses in [0, W] against alpha_def."""
    return empirical_bernstein(np.asarray(wlosses, dtype=float), alpha_def, rng=W)
