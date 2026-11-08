"""Code for the CDC 2025 paper: utilitarian vs sufficientarian upgrades of Grenoble's cycling network.

Modules
-------
config    all parameters of the paper (lambda values, eta, epsilon, costs, budgets, ...)
model     lower level: traffic factor, cyclist distance, AON flows, path and total exposure
justice   sufficientarian measures: path safety, thresholds, insufficiency I_p, I_r, J_I
costs     Cerema upgrade costs, B_max, km per DoS
mlcc      Maximize Largest Connected Component (Algorithm 2)
ga        genetic algorithm with efficient budget spending: GA-U and GA-S (Algorithm 1)
runner    one optimisation run per call (used to run several in parallel)
plotting  figures of the paper
utils     saving / loading results
"""
from .model import Network  # noqa: F401
