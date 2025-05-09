# 3B_lfunction_overlay_full.py
# Compares synthetic L_sym(s) from field-based resonance with classical L(E, s)
# for elliptic curve 37a1 (rank 1), using extended Euler product and real entropy data

import numpy as np
import matplotlib.pyplot as plt
import json
from math import isqrt

#~--------------------------------------------------------------------------#
# Prime generation + a_p table for E: y^2 + y = x^3 - x (LMFDB 37a1)
#~--------------------------------------------------------------------------#
def primes_upto(n):
    sieve = np.ones(n + 1, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(n**0.5) + 1):
        if sieve[p]:
            sieve[p*p:n+1:p] = False
    return [int(p) for p in np.nonzero(sieve)[0]]

# Extended a_p table for 37a1, generated via Sage
with open("ap_37a1.json", "r") as f:
    ap_dict = json.load(f)

def a_p(p):
    return ap_dict.get(str(p), 0)

#~--------------------------------------------------------------------------#
# Classical L-function (Euler product)
#~--------------------------------------------------------------------------#
def classical_L(s_vals, primes, a_ps):
    L_vals = []
    log_primes = np.log(primes)
    for s in s_vals:
        p_pow_s    = np.exp(-s * log_primes)
        p_pow_2s_1 = np.exp((1 - 2*s) * log_primes)
        factors    = 1 - a_ps * p_pow_s + p_pow_2s_1
        L_vals.append(1.0 / np.prod(factors))
    return np.array(L_vals, dtype=float)

#~--------------------------------------------------------------------------#
# User configuration
#~--------------------------------------------------------------------------#
OUT_FIG = "fig_Lsym_vs_classical_37a1.pdf"
s_grid  = np.linspace(0.5, 2.0, 400)
N_PRIMES = 256

#~ Load entropy → resonance weights
with open("entropy_by_prime.json", "r") as f:
    entropy_dict = json.load(f)

prime_set = sorted(int(p) for p in entropy_dict.keys())[:32]  # Top 32 primes with entropy
resonance_weights = [1 / entropy_dict[str(p)] for p in prime_set]

#~ Build synthetic L_sym(s)
L_sym_vals = [
    sum(R * np.exp(-p**s) for R, p in zip(resonance_weights, prime_set))
    for s in s_grid
]

#~ Build classical L(E, s)
primes = primes_upto(1500)[:N_PRIMES]
a_p_vals = np.array([a_p(p) for p in primes], dtype=float)
L_class_vals = classical_L(s_grid, primes, a_p_vals)

#~ Plot
plt.semilogy(s_grid, np.abs(L_class_vals), label="classical $L(E,s)$ (37a1)")
plt.semilogy(s_grid, np.abs(L_sym_vals), label="$L_{\text{sym}}(s)$ (resonance field)")
plt.axvline(1.0, ls="--", lw=0.8, c="k")
plt.xlabel("$s$"); plt.ylabel("$|L|$")
plt.title("Synthetic vs. Classical L-functions (Rank 1: 37a1)")
plt.legend(frameon=False)
plt.tight_layout(); plt.savefig(OUT_FIG, dpi=300)
print(f"Saved overlay figure →  {OUT_FIG}")

#~ Console report at s = 1
from scipy.interpolate import interp1d
print(f"L_sym(1)   ≈ {interp1d(s_grid, L_sym_vals)(1.0): .3e}")
print(f"L_class(1) ≈ {interp1d(s_grid, L_class_vals)(1.0): .3e}")
