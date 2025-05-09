# FC5: Symbolic Inverse Triplet Detection

# Generated from FC3 - Farey Analysis
triplets = [(3, 5, 7), (3, 5, 9), (3, 7, 9), (7, 5, 3), (9, 5, 3), (9, 7, 3)]

# Convert to set for fast lookup
triplet_set = set(triplets)

# Find inverse pairs: (i, j, k) <-> (k, j, i)
inverse_pairs = []
for i, j, k in triplets:
    if (k, j, i) in triplet_set:
        inverse_pairs.append(((i, j, k), (k, j, i)))

# Remove duplicates
unique_inverse_pairs = set(tuple(sorted(pair)) for pair in inverse_pairs)

# Display results
print("Symbolic Inverse Triplets:")
for pair in sorted(unique_inverse_pairs):
    print(f"{pair[0]}  <-->  {pair[1]}")

print(f"\nFound {len(unique_inverse_pairs)} inverse pairs out of {len(triplets)} total triplets.")
