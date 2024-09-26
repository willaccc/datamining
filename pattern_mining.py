# -*- coding: utf-8 -*-
# author: Willa Cheng

# this file includes functions caclulating similarities and distance 
# based on chapter 6 of book Data Mining by Han, Kamber and Pei

import numpy as np
from itertools import combinations
from collections import defaultdict

def has_infrequent_subset(candidate, frequent_itemsets):
    """Check if the candidate has any infrequent subsets.

    Parameters:
        candidate (frozenset): The candidate itemset to check.
        frequent_itemsets (list): List of frequent itemsets.

    Returns:
        bool: True if any subset is infrequent, False otherwise.
    """
    for subset in combinations(candidate, len(candidate) - 1):
        if frozenset(subset) not in frequent_itemsets:
            return True
    return False

def apriori_gen(frequent_itemsets, k):
    """Generate candidate itemsets of size k from frequent itemsets of size k-1.

    Parameters:
        frequent_itemsets (list): List of frequent itemsets of size k-1.
        k (int): The size of the candidate itemsets to generate.

    Returns:
        set: A set of candidate itemsets of size k.
    """
    candidates = set()
    len_frequent = len(frequent_itemsets)
    
    for i in range(len_frequent):
        for j in range(i + 1, len_frequent):
            # Only join if the first k-2 items are the same
            l1 = list(frequent_itemsets[i])[:k-2]
            l2 = list(frequent_itemsets[j])[:k-2]
            if l1 == l2:
                new_candidate = frozenset(frequent_itemsets[i] | frequent_itemsets[j])
                candidates.add(new_candidate)
    
    return candidates

def apriori(D, min_sup):
    """Find frequent itemsets in the transaction database using the Apriori algorithm.

    Parameters:
        D (list of list): The transaction database, where each transaction is a list of items.
        min_sup (int): The minimum support count threshold.

    Returns:
        list: A list of frequent itemsets found in the database.
    """
    # Remove the numpy conversion
    num_transactions = len(D)
    
    # Step 1: Count support of single items
    unique_items = set(item for transaction in D for item in transaction)
    L = {item: sum(1 for transaction in D if item in transaction) for item in unique_items if sum(1 for transaction in D if item in transaction) >= min_sup}
    
    frequent_itemsets = list(L.keys())
    all_frequent_itemsets = [frozenset([item]) for item in frequent_itemsets]

    k = 2
    while True:
        candidates = apriori_gen(all_frequent_itemsets, k)
        candidates = {candidate for candidate in candidates if not has_infrequent_subset(candidate, all_frequent_itemsets)}
        
        candidate_count = {candidate: 0 for candidate in candidates}
        
        for transaction in D:
            for candidate in candidates:
                if candidate.issubset(transaction):
                    candidate_count[candidate] += 1

        L_k = {itemset for itemset, count in candidate_count.items() if count >= min_sup}
        if not L_k:
            break

        all_frequent_itemsets.extend(L_k)
        frequent_itemsets = list(L_k)
        k += 1

    return all_frequent_itemsets

# Frequent Pattern Growth FP_growth method
class FPTreeNode:
    def __init__(self, item=None):
        """Initialize a node in the FP-tree.

        Paramters:
            item (str): The item name for this node.
        """
        self.item = item
        self.count = 1
        self.children = {}
        self.parent = None
        self.node_link = None

    def increment(self):
        """Increment the count of this node by 1."""
        self.count += 1

class FPTree:
    def __init__(self, transactions, min_sup):
        """Initialize the FP-tree and build it from the transaction data.

        Paramters:
            transactions (list of list): The transaction database.
            min_sup (int): The minimum support count threshold.
        """
        self.min_sup = min_sup
        self.header_table = {}
        self.root = FPTreeNode('null')
        self.build_tree(transactions)

    def build_tree(self, transactions):
        """Build the FP-tree from the transaction database.

        Paramters:
            transactions (list of list): The transaction database.
        """
        # Step 1: Count item frequencies
        item_count = defaultdict(int)
        for transaction in transactions:
            for item in transaction:
                item_count[item] += 1

        # Filter out items not meeting min_sup
        item_count = {item: count for item, count in item_count.items() if count >= self.min_sup}
        
        # Sort items by frequency
        sorted_items = sorted(item_count.items(), key=lambda x: x[1], reverse=True)
        sorted_items = [item[0] for item in sorted_items]

        # Step 2: Build the FP-tree
        for transaction in transactions:
            filtered_items = [item for item in sorted_items if item in transaction]
            if filtered_items:
                self.insert_tree(filtered_items)

    def insert_tree(self, items):
        """Insert a list of items into the FP-tree.

        Paramters:
            items (list): The list of frequent items to insert into the tree.
        """
        current_node = self.root
        for item in items:
            if item in current_node.children:
                current_node.children[item].increment()
            else:
                new_node = FPTreeNode(item)
                current_node.children[item] = new_node
                new_node.parent = current_node
                
                # Update header table
                if item not in self.header_table:
                    self.header_table[item] = []
                self.header_table[item].append(new_node)
            current_node = current_node.children[item]

    def mine_tree(self, prefix):
        """Mine the FP-tree for frequent itemsets.

        Paramters:
            prefix (list): The current prefix for the frequent itemsets.

        Returns:
            list: A list of tuples containing frequent itemsets and their support counts.
        """
        frequent_itemsets = []
        for item, nodes in sorted(self.header_table.items(), key=lambda x: x[0]):
            # Calculate support for the item
            support = sum(node.count for node in nodes)
            if support >= self.min_sup:
                new_prefix = prefix + [item]
                frequent_itemsets.append((new_prefix, support))
                
                # Construct conditional pattern base
                conditional_patterns = []
                for node in nodes:
                    path = []
                    self.get_prefix_path(node, path)
                    if path:
                        conditional_patterns.extend(path)
                
                # Build conditional FP-tree
                conditional_tree = FPTree(conditional_patterns, self.min_sup)
                # Mine the conditional tree
                frequent_itemsets.extend(conditional_tree.mine_tree(new_prefix))
        return frequent_itemsets
    
    def get_prefix_path(self, node, path):
        """Get the prefix path for a given node.

        Paramters:
            node (FPTreeNode): The node for which to retrieve the prefix path.
            path (list): The list to which the prefix path will be added.
        """
        if node.parent and node.parent.item != 'null':
            path.append(node.parent.item)
            self.get_prefix_path(node.parent, path)

def fp_growth(D, min_sup):
    """Find frequent itemsets using the FP-Growth algorithm.

    Paramters:
        D (list of list): The transaction database, where each transaction is a list of items.
        min_sup (int): The minimum support count threshold.

    Returns:
        list: A list of tuples containing frequent itemsets (as lists) and their support counts.
    """
    # Create FP-tree
    tree = FPTree(D, min_sup)
    # Mine the FP-tree for frequent itemsets
    return tree.mine_tree([])