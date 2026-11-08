"""
si_cycling - multi-modal, justice-oriented bike-network design (T-ITS special issue).

Extension of ITSC_2025 (car user equilibrium, capacity loss) and CDC_2025 (sufficientarian
upper level) with a mode-choice model and a machine-learning surrogate of the car UE.

Modules
-------
config      all parameters
trees       fast shortest-path-tree utilities (flow loading, path sums)
bike        bicycle layer (cyclist distance, AON flows, exposure J_E)       [= ITSC_2025]
cars        car layer (BPR, conjugate Frank-Wolfe user equilibrium)        [= ITSC_2025]
coupling    capacity rule theta = theta0 (1 - omega z / n)
modeshift   binary logit bike / car on the mean cyclist distance
surrogate   PCA + MLP surrogate of the UE flows + flow-conservation projection
justice     path sufficiency J_SI and trip insufficiency J_TI^b, J_TI^c
problems    evaluation chain and the utilitarian / sufficientarian objectives
mlcc        connectivity heuristic (warm start of the GA)                  [= CDC_2025]
ga, costs   genetic algorithm with Algorithm 1 decoding, Cerema costs      [= ITSC_2025]
runner, utils, plotting
"""
