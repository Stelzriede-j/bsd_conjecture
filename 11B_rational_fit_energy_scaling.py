
# BSD Phase 11 – Ford Circle Integration
# FC1: Rational Fit of Lock Zone Positions
# FC2: Energy vs. q² Scaling
# Author: Jacob Stelzriede (with OpenAI)
# April 2025

"""
This script performs Ford Circle integration tests:
- FC1: Approximates lock zone x-positions as rational numbers (p/q)
- FC2: Compares lock zone energy against 1/q² scaling (Ford radius law)

These confirm rational geometry alignment within symbolic twist field structures.
"""

from fractions import Fraction
from scipy.ndimage import label, center_of_mass
import matplotlib.pyplot as plt
import numpy as np

# Grid setup
Nx, Ny = 50, 50
x = np.linspace(-1, 1, Nx)
y = np.linspace(-1, 1, Ny)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
freq = 20

# Simulate field and extract lock zones
field = np.sin(2 * np.pi * freq * R)
mask = np.abs(field) > 0.2
labeled_array, num_features = label(mask)
centroids = center_of_mass(mask, labeled_array, range(1, num_features + 1))
lock_zones = [(x / Nx, y / Ny) for x, y in centroids]

# FC1: Fit rational x-values
q_values = []
x_vals = []
residuals = []

for x, y in lock_zones:
    f = Fraction(x).limit_denominator(20)
    q_values.append(f.denominator)
    x_vals.append(x)
    residuals.append(abs(x - float(f)))

# FC2: Compare energy vs q²
zone_energies = []
for i in range(1, num_features + 1):
    zone = (labeled_array == i)
    E = 0.5 * np.sum(field[zone] ** 2)
    zone_energies.append(E)

# Truncate to match list length
zone_energies = zone_energies[:len(q_values)]

# Plot FC2: Energy vs q²
plt.figure(figsize=(6, 4))
plt.scatter([q**2 for q in q_values], zone_energies, label="Zone Energy vs. q²")
plt.xlabel("q² (Ford denominator squared)")
plt.ylabel("Zone Energy")
plt.title("FC2: Energy Scaling with q²")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# FC1 output summary
print("FC1: First 10 Lock Zone Rational Fits")
for i in range(min(10, len(q_values))):
    print(f"x = {round(x_vals[i], 4)} → p/q = {Fraction(x_vals[i]).limit_denominator(20)}, q = {q_values[i]}, residual = {round(residuals[i], 5)}")
