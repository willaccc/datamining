# this is the testing case file for patter_mining.py methods
from pattern_mining import *
# Define your input data
D = [
    ['milk', 'bread'],
    ['bread', 'diaper', 'beer', 'egg'],
    ['milk', 'diaper', 'beer', 'cola'],
    ['milk', 'bread', 'diaper', 'beer'],
    ['bread', 'milk', 'diaper', 'cola']
]
min_sup = 2

# Run the Apriori algorithm
apriori_results = apriori(D, min_sup)
# Run the FP-Growth algorithm
fp_growth_results = fp_growth(D, min_sup)

# Output the results
print("Apriori Results:")
print(apriori_results)

print("\nFP-Growth Results:")
print(fp_growth_results)

# Original items in your dataset
original_items = {'beer', 'milk', 'diaper', 'cola', 'bread', 'egg'}

# Assuming `fp_growth_results` gives you counts
item_counts = {tuple(itemset): count for itemset, count in fp_growth_results}

# Create frozensets for all frequent itemsets based on counts
fp_growth_frozensets = set()
for itemset, count in item_counts.items():
    if count >= min_sup:  # Check against your min_support
        fp_growth_frozensets.add(frozenset(itemset))

# Include combinations of frequent items based on their counts
for r in range(1, len(original_items) + 1):
    for combo in combinations(original_items, r):
        if tuple(combo) in item_counts:
            fp_growth_frozensets.add(frozenset(combo))

# Convert Apriori output to frozensets
apriori_frozensets = {frozenset(itemset) for itemset in apriori_results}

# Now you can compare the two sets
are_equal = apriori_frozensets == fp_growth_frozensets

# Output the results
print("Apriori Results (frozensets):", apriori_frozensets)
print("FP-Growth Results (frozensets):", fp_growth_frozensets)
print("Do they match?", are_equal)