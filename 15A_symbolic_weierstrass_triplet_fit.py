# 15a_symbolic_weierstrass_triplet_fit.py
# Test whether symbolic group behavior emerges from centroids using Weierstrass addition

# Run 6BC to generate centroids.npy

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

coords = np.load("centroids.npy")
coords = coords[:, ::-1]  # flip y, x → x, y
coords = coords / np.max(coords, axis=0)  # normalize to [0, 1]

print(f"Loaded {len(coords)} centroids")

#~ Fit Weierstrass curve y^2 = x^3 + ax + b
x_vals = coords[:, 0]
y_vals = coords[:, 1]
X = np.vstack([x_vals**3, x_vals, np.ones_like(x_vals)]).T
y_sq = y_vals**2
coeffs, *_ = np.linalg.lstsq(X, y_sq, rcond=None)
a, b = coeffs[1], coeffs[2]

print(f"Fitted curve: y^2 = x^3 + ({a:.3f})x + ({b:.3f})")

#~ Define Weierstrass addition (affine case)
def weierstrass_add(P, Q, a):
    x1, y1 = P
    x2, y2 = Q
    if np.allclose(P, Q):
        m = (3 * x1**2 + a) / (2 * y1)
    else:
        m = (y2 - y1) / (x2 - x1)
    x3 = m**2 - x1 - x2
    y3 = m * (x1 - x3) - y1
    return np.array([x3, y3])

#~ Test all centroid pairs for Weierstrass addition closure

match_count = 0
close_triplets = []
tolerance = 0.15

for i in range(len(coords)):
    for j in range(i + 1, len(coords)):
        try:
            P, Q = coords[i], coords[j]
            if np.isclose(P[0], Q[0]) and not np.allclose(P, Q):
                continue  # vertical slope, invalid addition
            R_pred = weierstrass_add(P, Q, a)
            dists = cdist([R_pred], coords)[0]
            k = np.argmin(dists)
            if dists[k] < tolerance:
                match_count += 1
                close_triplets.append((i, j, k))
                print(f"g_{i} + g_{j} ≈ g_{k} → dist = {dists[k]:.3f}")
        except Exception:
            continue

print(f"Total Weierstrass-compatible triplets: {match_count}")

#~ Plot curve and centroids
x_plot = np.linspace(min(coords[:, 0]), max(coords[:, 0]), 400)
y_plot = np.sqrt(x_plot**3 + a*x_plot + b)

plt.figure(figsize=(6, 6))
plt.scatter(coords[:, 0], coords[:, 1], color='red', label='Centroids')
plt.plot(x_plot, y_plot, label='y = +sqrt(x^3 + ax + b)', color='blue')
plt.plot(x_plot, -y_plot, label='y = -sqrt(x^3 + ax + b)', color='blue', linestyle='--')
plt.title("Fitted Elliptic Curve and Symbolic Centroids")
plt.grid(True)
plt.legend()
plt.show()
