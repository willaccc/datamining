from pattern_mining import *
D = [
    ['milk', 'bread'],
    ['bread', 'diaper', 'beer', 'egg'],
    ['milk', 'diaper', 'beer', 'cola'],
    ['milk', 'bread', 'diaper', 'beer'],
    ['bread', 'milk', 'diaper', 'cola'], 
    ['apple', 'banana', 'cherry'], 
    ['water', 'banana', 'cola']
]

min_sup = 2
# Example debug output for counting frequencies
# def count_item_frequencies(transactions):
#     freq = {}
#     for transaction in transactions:
#         for item in transaction:
#             if item in freq:
#                 freq[item] += 1
#             else:
#                 freq[item] = 1
#     print("Item Frequencies:", freq)
#     return freq

# # Call this before building the FP-tree
# frequencies = count_item_frequencies(D)

# filtered_items = [item for item, count in frequencies.items() if count >= min_sup]
# print("Filtered Items:", filtered_items)
fp_growth(D, min_sup)