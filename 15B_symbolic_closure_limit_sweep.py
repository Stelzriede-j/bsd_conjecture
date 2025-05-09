# 15b_symbolic_closure_limit_sweep.py
# Sweep epsilon values to observe symbolic closure behavior under Weierstrass addition

import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

coords = np.load("centroids.npy")
coords = coords[:, ::-1]  # y, x → x, y
coords = coords / np.max(coords, axis=0)

print(f"Loaded {len(coords)} centroids")

# Fit Weierstrass curve: y^2 = x^3 + ax + b
x_vals = coords[:, 0]
y_vals = coords[:, 1]
X = np.vstack([x_vals**3, x_vals, np.ones_like(x_vals)]).T
y_sq = y_vals**2
coeffs, *_ = np.linalg.lstsq(X, y_sq, rcond=None)
a, b = coeffs[1], coeffs[2]

print(f"Fitted curve: y^2 = x^3 + ({a:.3f})x + ({b:.3f})")

# Define Weierstrass addition

def weierstrass_add(P, Q, a):
    x1, y1 = P
    x2, y2 = Q
    if np.isclose(x1, x2) and not np.allclose(P, Q):
        return None  # vertical slope case
    if np.allclose(P, Q):
        m = (3 * x1**2 + a) / (2 * y1)
    else:
        m = (y2 - y1) / (x2 - x1)
    x3 = m**2 - x1 - x2
    y3 = m * (x1 - x3) - y1
    return np.array([x3, y3])

# Sweep epsilon
sweep_eps = np.linspace(0.05, 0.3, 26)
match_counts = []

for epsilon in sweep_eps:
    count = 0
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            P, Q = coords[i], coords[j]
            R_pred = weierstrass_add(P, Q, a)
            if R_pred is None:
                continue
            dists = cdist([R_pred], coords)[0]
            if np.min(dists) < epsilon:
                count += 1
    match_counts.append(count)
    print(f"Epsilon {epsilon:.3f}: {count} matches")

# Plot results
plt.figure(figsize=(7, 5))
plt.plot(sweep_eps, match_counts, marker='o')
plt.xlabel("Closure tolerance epsilon")
plt.ylabel("Weierstrass-compatible triplets")
plt.title("Symbolic Closure vs. Epsilon Sweep")
plt.grid(True)
plt.tight_layout()
plt.show()
