#!/usr/bin/env python3
# 3c_tunable_overlay.py
# ------------------------------------------------------------
# Synthetic L_sym(s)   vs   Classical L(E,s)  (tunable curve)
# – No Sage / PARI / network needed
# – Caches a_p(p) to JSON for repeatability
# ------------------------------------------------------------

import os, json, warnings, math, numpy as np, matplotlib.pyplot as plt
from mpmath import mp
from scipy.interpolate import interp1d

# ========= 1. Tunable parameters ==========================================
# Curve in short Weierstrass form  y^2 = x^3 + A x + B
CURVE_LABEL  = "5077a1"        # used in filenames only
A, B         = -480, -7776     # coefficients for 5077a1
CONDUCTOR_N  = 5077
ROOT_NUMBER  = -1              # w = -1  (check LMFDB if unsure)
RANK_LABEL   = "rank-3"        # used in plot title / caption
JSON_AP_FILE = f"ap_{CURVE_LABEL}.json"
PRIME_MAX    = 2999            # primes up to this used in Euler sums
N_TERMS      = 12000           # Dirichlet terms in functional eq.
MP_PREC      = 60              # decimal digits for mpmath


# ---------- L_sym parameters: load spectral‐entropy weights ---------------

# L_sym parameters (same entropy weights you used before)
#PRIME_ENTROPY = {13: 7.46, 17: 7.48, 19: 7.49}

ENTROPY_FILE = "entropy_by_prime.json"   # adjust path if needed
try:
    PRIME_ENTROPY = {int(p): float(h) for p, h in
                     json.load(open(ENTROPY_FILE)).items()}
except (FileNotFoundError, json.JSONDecodeError) as e:
    warnings.warn(f"[warn] {ENTROPY_FILE} not found or invalid; "
                  "falling back to default trio {13,17,19}.")
    PRIME_ENTROPY = {13: 7.46, 17: 7.48, 19: 7.49}


# ========= 2. Helper functions ===========================================
mp.dps = MP_PREC

def sieve_primes(n: int):
    sieve = bytearray(b"\x01") * (n + 1)
    sieve[:2] = b"\x00\x00"
    for p in range(2, int(n**0.5) + 1):
        if sieve[p]:
            sieve[p*p:n+1:p] = b"\x00" * len(range(p*p, n + 1, p))
    return [i for i in range(2, n + 1) if sieve[i]]

def legendre(a: int, p: int) -> int:
    return pow(a % p, (p - 1) // 2, p)

def point_count_ap(p: int) -> int:
    """Exact a_p for general model y^2 + a1 xy + a3 y = x^3 + a2 x^2 + a4 x + a6."""
    a1, a2, a3, a4, a6 = 1, 0, 0, -480, -7776   # 5077a1
    if p in (2, CONDUCTOR_N):      # skip bad primes
        return 0
    npts = 1  # point at infinity
    for x in range(p):
        rhs = (x**3 + a2*x**2 + a4*x + a6) % p
        for y in range(p):
            lhs = (y*y + a1*x*y + a3*y) % p
            if lhs == rhs:
                npts += 1
    return p + 1 - npts

# ========= 3. a_p cache (JSON) ===========================================
if os.path.exists(JSON_AP_FILE):
    ap_dict = json.load(open(JSON_AP_FILE))
else:
    primes = sieve_primes(PRIME_MAX)
    ap_dict = {str(p): point_count_ap(p) for p in primes}
    json.dump(ap_dict, open(JSON_AP_FILE, "w"))
    print(f"[cache] wrote {JSON_AP_FILE}")

def a_p(p: int) -> int:
    return ap_dict.get(str(p), 0)

# ========= 4. Synthetic  L_sym(s) ========================================
# ---------- s‑grid that avoids all integer poles (1, 2, …) ----------------
left  = np.linspace(0.5, 0.99, 200, endpoint=False)   # < 1
right = np.linspace(1.01, 2.0, 200, endpoint=False)   # < 2
s_grid = np.concatenate((left, right))

entropy_weights = [1 / PRIME_ENTROPY[p] for p in PRIME_ENTROPY]
L_sym = np.array([
    sum(w * math.exp(-p**s) for w, p in zip(entropy_weights, PRIME_ENTROPY))
    for s in s_grid
])

# ========= 5. Classical  L(E,s)  via functional eq. ======================
def multiplicative_an(N: int):
    """Return list a[1..N] using a_p recursion."""
    a = [mp.mpf(0)] * (N + 1)
    a[1] = mp.mpf(1)
    for n in range(2, N + 1):
        # factor n = p^k * m
        m, k, p = n, 0, None
        # find first prime divisor
        for q in (2, *range(3, int(math.sqrt(n)) + 1, 2)):
            if m % q == 0:
                p = q
                break
        if p is None:
            p = m
        while m % p == 0:
            m //= p
            k += 1
        if k == 1:
            a[n] = a_p(p) * a[n // p]
        else:
            a[n] = a_p(p) * a[n // p] - p * a[n // (p * p)]
    return a


def build_a_coeffs(N):
    """Return list a[1..N] using complete Euler recursion."""
    ap_cache = {p: mp.mpf(a_p(p)) for p in sieve_primes(PRIME_MAX)}
    a = [mp.mpf(0)] * (N + 1)
    a[1] = mp.mpf(1)

    def a_n(n):
        if a[n] != 0:                 # memoised
            return a[n]
        # split off smallest prime
        for p in (2, *range(3, int(math.sqrt(n))+1, 2)):
            if n % p == 0:
                break
        else:
            p = n                     # n itself prime
        k, m = 0, n
        while m % p == 0:
            m //= p; k += 1
        if m == 1:                    # pure p^k
            if k == 1:
                a[n] = ap_cache[p]
            else:
                a[n] = ap_cache[p] * a_n(n//p) - p * a_n(n//(p*p))
        else:                         # multiplicative split
            a[n] = a_n(p**k) * a_n(m)
        return a[n]

    for n in range(2, N+1):
        a_n(n)
    return a

print("[info] computing a_n up to", N_TERMS)
#a_coeffs = multiplicative_an(N_TERMS)
a_coeffs = build_a_coeffs(N_TERMS)
print("a_2 =", a_coeffs[2], "   a_3 =", a_coeffs[3],
      "   a_6 =", a_coeffs[6], "   a_11 =", a_coeffs[11])


def L_classical(s):
    term1 = mp.fsum(a_coeffs[n] / mp.power(n, s) for n in range(1, N_TERMS + 1))
    X = (mp.sqrt(CONDUCTOR_N) / (2 * mp.pi)) ** (1 - 2 * s) * mp.gamma(1 - s) / mp.gamma(s)
    term2 = ROOT_NUMBER * X * mp.fsum(a_coeffs[n] / mp.power(n, 1 - s)
                                      for n in range(1, N_TERMS + 1))
    return term1 + term2

print("[info] evaluating classical L on grid")
EPS = mp.mpf('1e-8')          #   1 × 10⁻⁸  is plenty
L_class = np.array([
    float(L_classical(mp.mpf(str(s)) + (EPS if abs(s-1.0) < 1e-10 else 0)))
    for s in s_grid
])

# ========= 6. Output CSVs and figure =====================================
np.savetxt("Lsym_data.csv",   np.column_stack([s_grid, L_sym]),   delimiter=",")
np.savetxt("Lclass_data.csv", np.column_stack([s_grid, L_class]), delimiter=",")

fig_name = f"fig_Lsym_vs_classical_{CURVE_LABEL}.pdf"
plt.semilogy(s_grid, np.abs(L_class), label=f"classical $L(E,s)$ ({CURVE_LABEL})")
plt.semilogy(s_grid, np.abs(L_sym),   label="$L_{\\text{sym}}(s)$")
plt.axvline(1.0, ls="--", lw=0.8, c="k")
plt.xlabel("$s$"); plt.ylabel("$|L|$")
plt.title(f"Synthetic vs Classical L ( {RANK_LABEL} : {CURVE_LABEL} )")
plt.legend(frameon=False)
plt.tight_layout(); plt.savefig(fig_name, dpi=300)

# console diagnostics at s ≈ 1
idx_lo = np.max(np.where(s_grid < 1.0))
idx_hi = idx_lo + 1
print(f"L_sym(≈1)  ≈ {0.5*(L_sym[idx_lo]+L_sym[idx_hi]):.3e}")
print(f"L_class(≈1)≈ {0.5*(L_class[idx_lo]+L_class[idx_hi]):.3e}")
print(f"[done] saved plot  →  {fig_name}")
