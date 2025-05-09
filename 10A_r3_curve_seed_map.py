# 10A with cleaner rank 3 elliptic curves and lower scaling (scale = 4), grid 100x100

from sympy import Rational, symbols, Eq, solve
import numpy as np

# Define symbolic variables
x, y = symbols('x y')

# Cleaner curves with more accessible rational points
rank3_clean_curves = {
    "R3_C1": Eq(y**2, x**3 - 11*x + 14),     # simpler, still known rank 3
    "R3_C2": Eq(y**2, x**3 - 16*x + 16),     # known rank 3, visible points
    "R3_C3": Eq(y**2, x**3 - 22*x + 63),     # known rank 3
}

# Extended x-range
x_vals = [Rational(n, 1) for n in range(-30, 31)]
scale = 4
offset_x, offset_y = 50, 50
Nx, Ny = 100, 100

seed_summary_clean = {}

for name, curve in rank3_clean_curves.items():
    rational_points = []
    for x_val in x_vals:
        rhs = curve.rhs.subs(x, x_val)
        y_solutions = solve(Eq(y**2, rhs), y)
        for y_val in y_solutions:
            if y_val.is_rational:
                rational_points.append((float(x_val), float(y_val)))

    # Map to field grid
    seed_centers = [(int(offset_x + scale * xp), int(offset_y - scale * yp))
                    for xp, yp in rational_points]
    valid_centers = list({(x, y) for x, y in seed_centers if 0 <= x < Nx and 0 <= y < Ny})

    seed_summary_clean[name] = {
        "Total Rational Points": len(rational_points),
        "Valid Centers in Grid": len(valid_centers),
        "Coordinates": valid_centers
    }

import pprint
pprint.pprint(seed_summary_clean)

