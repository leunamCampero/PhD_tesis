"""
itsc_cycling - safety-oriented bike-lane upgrades with car travel-time trade-off (ITSC 2025).

Modules
-------
config    all parameters
trees     fast shortest-path-tree utilities (flow loading, path sums) used by both layers
bike      bicycle layer: cyclist distance, all-or-nothing flows, exposure J_E
cars      car layer: BPR, all-or-nothing loading, conjugate Frank-Wolfe user equilibrium
coupling  which car roads lose capacity when bike arcs are upgraded
costs     Cerema costs, budgets
ga        genetic algorithm (priority encoding + Algorithm 1 decoding)
problem   the trade-off objective of Problem 2 and its normalisation
runner    one optimisation per call (parallel runs)
plotting, utils
"""
