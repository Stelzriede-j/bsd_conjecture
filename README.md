# BSD Symbolic Field Suite

This repository contains a complete simulation and validation suite for constructing and analyzing symbolic analogs of the Birch–Swinnerton–Dyer (BSD) Conjecture using nonlinear twist-compression fields.

The framework is designed to:

* Evolve scalar fields with twist injections and nonlinear compression.
* Detect symbolic lock zones and construct group-like triplet operations.
* Compute entropy-based spectral resonance and build synthetic L-functions.
* Validate BSD analog components: rank, torsion, regulator, Tamagawa factors, and $\Sha$.

All results are reproducible and organized by phase.

---

## Installation & Requirements

* Python 3.8+
* NumPy, SciPy, Matplotlib

Install dependencies:

```bash
pip install numpy scipy matplotlib
```

---

## Phase Summary and Script Index

### Phase 2: Field Evolution and Analysis

| Script                         | Description                                                               |
| ------------------------------ | ------------------------------------------------------------------------- |
| `2A_lock_zone_detection.py`    | Detects symbolic lock zones using mod-$p$ projection.                     |
| `2B_twist_lock_analysis.py`    | Evolves the field with configurable injections; saves `phi.npy`.          |
| `2C_field_entropy.py`          | Computes spectral entropy $H_p$ for various primes; exports JSON weights. |
| `2D_entropy_parameter_scan.py` | Sweeps $(\lambda, n)$ to analyze symbolic lock-in conditions.             |

### Phase 3: Synthetic L-function Construction

| Script                         | Description                                                              |
| ------------------------------ | ------------------------------------------------------------------------ |
| `3A_lfunction_exp_decay.py`    | Constructs synthetic $L_{\text{sym}}(s)$ from field entropy.             |
| `3B_lfunction_overlay_full.py` | Compares synthetic L-function to classical $L(E, s)$ (e.g., curve 37a1). |

### Phase 6: Symbolic Triplet Extraction and Closure

| Script                              | Description                                                |
| ----------------------------------- | ---------------------------------------------------------- |
| `6A_full_twist_field_mst.py`        | Builds MST over centroid graph.                            |
| `6BC_symbolic_triplets.py`          | Extracts symbolic triplets via geometric closure.          |
| `6D_symbolic_group_closure_test.py` | Tests group closure relations in symbolic triplets.        |
| `6E_symbolic_conflict_sweep.py`     | Measures triplet conflict rate as a function of frequency. |

### Phase 7–8: Elliptic Echo and Rank Estimation

| Script                         | Description                                    |
| ------------------------------ | ---------------------------------------------- |
| `7_master_curve_projection.py` | Fits Weierstrass form to centroid cloud.       |
| `8A_curve_projection_sweep.py` | Sweeps frequency and logs curve-fit stability. |
| `8B_rank_spectrum_sweep.py`    | Measures vanishing order and symbolic rank.    |

### Phase 9: Torsion Analysis

| Script                    | Description                                           |
| ------------------------- | ----------------------------------------------------- |
| `9A_torsion_structure.py` | Detects self-inverse symbolic triplets.               |
| `9B_torsion_response.py`  | Analyzes fixed-point torsion under mod-p projections. |
| `9C_torsion_signature.py` | Logs torsion symmetry and cyclic substructure.        |

### Phase 10: High-Rank Symbolic Fields

| Script                      | Description                                            |
| --------------------------- | ------------------------------------------------------ |
| `10A_r3_curve_seed_map.py`  | Seeds known rank-3 curve and tracks symbolic recovery. |
| `10B_r3_resonance_field.py` | Tests resonance behavior under rank-3 symbolic seed.   |

### Phase 11: Rational Geometry (Ford Circle)

| Script                                | Description                                           |
| ------------------------------------- | ----------------------------------------------------- |
| `11A_fc1_rational_fit.py`             | Fits lock zones to Farey neighbors.                   |
| `11B_farey_neighbor_triplet_check.py` | Checks triplet overlap with Farey sequence adjacency. |
| `11C_farey_mst_edges.py`              | Analyzes MST edge rationality.                        |
| `11D_farey_inverse_symmetry.py`       | Tracks inverse triplets and rational layer symmetry.  |
| `11F_lockzone_q_density.py`           | Fits $1/q^2$ rational density for zone centers.       |

### Phase 12: Amplitude Sweep and Scaling Laws

| Script                          | Description                                        |
| ------------------------------- | -------------------------------------------------- |
| `12A_twist_amplitude_sweep.py`  | Sweeps twist amplitude and tracks symbolic rank.   |
| `12D_full_energy_extraction.py` | Extracts energy scaling law across symbolic zones. |

### Phase 13: BSD Term Construction

| Script                        | Description                                        |
| ----------------------------- | -------------------------------------------------- |
| `13A_symbolic_regulator.py`   | Computes log-distance regulator matrix.            |
| `13B_tamagawa_simulation.py`  | Simulates symbolic degradation as Tamagawa analog. |
| `13C_sha_triplet_detector.py` | Detects symbolic $\Sha$ via local-global mismatch. |

### Phase 14: Cymatic Analog Tests

| Script                               | Description                                                  |
| ------------------------------------ | ------------------------------------------------------------ |
| `14A_cymatic_pattern_visualizer.py`  | Visualizes nodal structure of fixed-frequency scalar fields. |
| `14B_cymatic_resonance_sweep.py`     | Detects cymatic lock-in at symbolic thresholds.              |
| `14C_cymatic_structure_emergence.py` | Builds radial lock zone structure plots.                     |
| `14D_cymatic_bsd_ladder.py`          | Confirms ladder emergence consistent with BSD structure.     |

### Phase 15: Symbolic Group Validation

| Script                                    | Description                                                            |
| ----------------------------------------- | ---------------------------------------------------------------------- |
| `15a_symbolic_weierstrass_triplet_fit.py` | Tests triplet closure under elliptic addition.                         |
| `15b_closure_limit_sweep.py`              | Sweeps epsilon to track symbolic convergence.                          |
| `15c_symbolic_weierstrass_group_law.py`   | Validates associativity $(g_i + g_j) + g_k \approx g_i + (g_j + g_k)$. |

---

## Output Files

| File                             | Description                                                           |
| -------------------------------- | --------------------------------------------------------------------- |
| `phi.npy`                        | Saved twist-compression field used for entropy and symbolic analysis. |
| `entropy_by_prime.json`          | Entropy values $H_p$ for use in synthetic L-function.                 |
| `ap_37a1.json`                   | a\_p table for elliptic curve 37a1 (rank 1).                          |
| `fig_Lsym_vs_classical_37a1.pdf` | Overlay of synthetic and classical L-functions.                       |
| `entropy_profile_plot.png`       | Spectral entropy per prime.                                           |

---

## Citation

This code supports:

> **"Resonance Fields and the Constructive Emergence of BSD Group Structures"**
> Jacob Stelzriede, 2025

---

## License

MIT License

For questions or feedback, contact: [jacob@stelzriede.org](mailto:jacob@stelzriede.org)
