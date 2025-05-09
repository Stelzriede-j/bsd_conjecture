# Phase 9C: Analyze torsion triplets from 9B sweep for cyclic or self-inverse behavior

import pandas as pd

# Manually bring in Phase 9B results (or read from CSV if available)
torsion_triplets_data = [
    {
        "Curve": "T1",
        "Triplets": ["g_2 + g_1 = g_3", "g_3 + g_1 = g_2"]
    },
    {
        "Curve": "T2",
        "Triplets": ["g_1 + g_2 = g_16", "g_16 + g_2 = g_1"]
    },
    {
        "Curve": "T3",
        "Triplets": []
    },
    {
        "Curve": "T4",
        "Triplets": ["g_3 + g_4 = g_6", "g_6 + g_4 = g_3"]
    },
    {
        "Curve": "T5",
        "Triplets": []
    }
]

# Analyze for symmetric inverses and cyclic folding
def analyze_triplets(triplet_list):
    triplet_set = set(triplet_list)
    inverse_pairs = 0
    fixed_ops = 0
    for t in triplet_list:
        left, right = t.split("=")
        left = left.strip()
        right = right.strip()
        g1, g2 = [s.strip() for s in left.split("+")]
        # Check reverse direction
        rev1 = f"{right} + {g2} = {g1}"
        rev2 = f"{right} + {g1} = {g2}"
        if rev1 in triplet_set or rev2 in triplet_set:
            inverse_pairs += 1
        if g1 == right or g2 == right:
            fixed_ops += 1
    return inverse_pairs, fixed_ops

results = []
for entry in torsion_triplets_data:
    curve = entry["Curve"]
    triplets = entry["Triplets"]
    inverse_pairs, fixed_ops = analyze_triplets(triplets)
    results.append({
        "Curve": curve,
        "Total Triplets": len(triplets),
        "Inverse Pairs": inverse_pairs,
        "Fixed Points": fixed_ops,
        "Torsion Signature": "Yes" if inverse_pairs > 0 or fixed_ops > 0 else "No"
    })


df_9c = pd.DataFrame(results)
print(df_9c)
df_9c.to_csv("phase9c_torsion_signature_log.csv", index=False)