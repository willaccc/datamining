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