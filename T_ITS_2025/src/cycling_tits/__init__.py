"""Implementation of the methodology of

M. Campero Jurado, C. Canudas-de-Wit, G. De Nunzio,
"A Graph-Based Approach to the Topological Optimization of Cycling Networks for
the Improvement of Safety and Comfort of Cyclists", IEEE T-ITS.
"""
from .data import CITIES, DATA, FIGURES, RESULTS, load_network, load_survey, pretty
from .model import Model, PRESETS, alphas, max_budget
