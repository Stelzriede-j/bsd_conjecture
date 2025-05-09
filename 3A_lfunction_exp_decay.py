
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Prime set and entropies
prime_set = [13, 17, 19]
approx_entropies = {
    13: 7.46,
    17: 7.48,
    19: 7.49
}
resonance_weights = [1 / approx_entropies[p] for p in prime_set]

# Exponential L-function with k=1.0
s_values = np.linspace(0.5, 2.0, 200)
L_exp = []
for s in s_values:
    val = sum(R * np.exp(-p**(1.0 * s)) for R, p in zip(resonance_weights, prime_set))
    L_exp.append(val)

# Interpolate for derivative tests
L_interp = interp1d(s_values, L_exp, kind='cubic', fill_value="extrapolate")

# Numerical derivative functions
def numerical_derivative(f, x, dx=1e-5, order=1):
    if order == 1:
        return (f(x + dx) - f(x - dx)) / (2 * dx)
    elif order == 2:
        return (f(x + dx) - 2*f(x) + f(x - dx)) / (dx ** 2)
    else:
        raise ValueError("Only first and second derivatives are supported.")

# Evaluate at s = 1
s_target = 1.0
L_value = L_interp(s_target)
L_prime = numerical_derivative(L_interp, s_target, order=1)
L_double_prime = numerical_derivative(L_interp, s_target, order=2)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(s_values, L_exp, color='teal')
plt.axhline(0, linestyle=':', color='black')
plt.title("Exponential L-function: L(s) = sum(R_p * exp(-p^s)) with k=1.0")
plt.xlabel("s")
plt.ylabel("L_field(s)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Output derivatives
print(f"L(1) = {L_value}")
print(f"L'(1) = {L_prime}")
print(f"L''(1) = {L_double_prime}")
