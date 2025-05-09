# 15c_symbolic_weierstrass_group_law.py
# Test symbolic associativity: (P + Q) + R ≈ P + (Q + R)

import numpy as np
from scipy.spatial.distance import cdist

coords = np.load("centroids.npy")
coords = coords[:, ::-1]  # flip y, x → x, y
coords = coords / np.max(coords, axis=0)

print(f"Loaded {len(coords)} centroids")

#~ Fit elliptic curve y^2 = x^3 + ax + b
x_vals = coords[:, 0]
y_vals = coords[:, 1]
X = np.vstack([x_vals**3, x_vals, np.ones_like(x_vals)]).T
y_sq = y_vals**2
coeffs, *_ = np.linalg.lstsq(X, y_sq, rcond=None)
a, b = coeffs[1], coeffs[2]

print(f"Fitted curve: y^2 = x^3 + ({a:.3f})x + ({b:.3f})")

#~ Define Weierstrass addition

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

#~ Associativity test: (P + Q) + R ≈ P + (Q + R)
tolerance = 0.15
passed = 0
attempts = 0

for i in range(len(coords)):
    for j in range(len(coords)):
        for k in range(len(coords)):
            if len(set([i, j, k])) < 3:
                continue
            P, Q, R = coords[i], coords[j], coords[k]
            A = weierstrass_add(P, Q, a)
            if A is None:
                continue
            L = weierstrass_add(A, R, a)
            if L is None:
                continue
            B = weierstrass_add(Q, R, a)
            if B is None:
                continue
            R_assoc = weierstrass_add(P, B, a)
            if R_assoc is None:
                continue
            dist = np.linalg.norm(L - R_assoc)
            if dist < tolerance:
                passed += 1
            attempts += 1

print(f"Associativity satisfied: {passed}/{attempts} cases (\u03B5 < {tolerance})")
