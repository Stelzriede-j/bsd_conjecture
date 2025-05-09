# 2C_entropy_from_field.py
# Computes spectral entropy H_p for a given twist-compression field phi.npy
# across a set of prime moduli and outputs entropy_by_prime.json

import numpy as np
import matplotlib.pyplot as plt
import json

#~--------------------------------------------------------------------------#
# Helper: compute entropy from normalized histogram
#~--------------------------------------------------------------------------#
def spectral_entropy(phi_mod_p, p):
    hist = np.bincount(phi_mod_p.flatten(), minlength=p)
    rho = hist / np.sum(hist)
    rho = rho[rho > 0]  # avoid log(0)
    H_p = -np.sum(rho * np.log2(rho))
    return H_p

#~--------------------------------------------------------------------------#
# Load field from phi.npy
#~--------------------------------------------------------------------------#
try:
    phi = np.load("phi.npy")
    print("Loaded field phi.npy with shape:", phi.shape)
except FileNotFoundError:
    print("Error: phi.npy not found. Please run field evolution first.")
    exit()

#~--------------------------------------------------------------------------#
# Prime moduli to test
#~--------------------------------------------------------------------------#
mod_p_values = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
Hp = {}

#~--------------------------------------------------------------------------#
# Compute spectral entropy for each prime
#~--------------------------------------------------------------------------#
for p in mod_p_values:
    phi_mod = np.round(phi * 1000).astype(int) % p
    H = spectral_entropy(phi_mod, p)
    Hp[p] = H
    print(f"H({p}) = {H:.4f}")

#~--------------------------------------------------------------------------#
# Save to JSON
#~--------------------------------------------------------------------------#
with open("entropy_by_prime.json", "w") as f:
    json.dump(Hp, f, indent=2)
print("Saved entropy_by_prime.json")

#~--------------------------------------------------------------------------#
# Optional: plot entropy profile
#~--------------------------------------------------------------------------#
plt.figure(figsize=(8, 4))
plt.plot(list(Hp.keys()), list(Hp.values()), marker='o', color='teal')
plt.xlabel("Prime modulus p")
plt.ylabel("Spectral entropy H_p")
plt.title("Spectral Entropy vs Prime Modulus")
plt.grid(True)
plt.tight_layout()
plt.savefig("entropy_profile_plot.png", dpi=300)
plt.show()