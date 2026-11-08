"""Reproduction code for the ECC 2025 paper on bi-level upgrading of Grenoble's cycling network.

Modules
-------
config     all parameters of the paper
model      traffic factor, cyclist distance, all-or-nothing flows, exposure J_E
costs      Cerema upgrade costs, budget B_max, km per DoS
mlcc       Maximize-Largest-Connected-Component algorithm (Algorithm 2)
gab        Genetic Algorithm with efficient Budget spending (pygmo SGA + Algorithm 1)
plotting   bar chart and maps of the paper
"""
from .model import Network  # noqa: F401
