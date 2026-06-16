"""Compute-Adaptive Risk Control (CARC) reference implementation."""
from .selector import select_risk, select_dual, select_shift, select_naive
from .simulate import EarlyExitSim, fit_weights_logistic
from .chain import build_chain, build_threshold_family
from . import pvalues, certify

__all__ = [
    "select_risk", "select_dual", "select_shift", "select_naive",
    "EarlyExitSim", "fit_weights_logistic", "build_chain", "build_threshold_family",
    "pvalues", "certify",
]
