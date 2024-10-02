# -*- coding: utf-8 -*-
# author: Willa Cheng

# this file includes functions caclulating similarities and distance 
# based on chapter 8 of book Data Mining by Han, Kamber and Pei

import numpy as np

class DecisionTree:
    """
    A decision tree classifier for both categorical and continuous attributes.
    
    Parameters:
        max_depth (int): The maximum depth of the tree.
        criterion (str): The criterion for splitting ('gini', 'information_gain', 'gain_ratio').
        binary_tree (bool): Whether to create a binary tree or allow multi-way splits for categorical attributes.
        tree (tuple): The constructed decision tree.
    """

    def __init__(self, max_depth=None, criterion='gini', binary_tree=True):
        """
        Initializes the DecisionTree with specified parameters.

        Parameters:
            max_depth (int, optional): The maximum depth of the tree. Defaults to None (no limit).
            criterion (str, optional): The criterion for splitting. Options are 'gini', 'information_gain', 'gain_ratio'. Defaults to 'gini'.
            binary_tree (bool, optional): Whether to create a binary tree. Defaults to True.
        """
        self.max_depth = self.auto_max_depth(max_depth)
        self.criterion = criterion
        self.binary_tree = binary_tree
        self.tree = None

    def auto_max_depth(self, max_depth):
        if max_depth is not None:
            return max_depth
        
        return 1 + int(np.log2(len(X))) if len(X) > 0 else None

    def fit(self, X, y):
        """
        Fits the decision tree to the training data.

        Parameters:
            X (np.ndarray): The feature matrix.
            y (np.ndarray): The target labels.
        """
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        """
        Recursively builds the decision tree.

        Parameters:
            X (np.ndarray): The feature matrix.
            y (np.ndarray): The target labels.
            depth (int): The current depth of the tree.

        Returns:
            tuple or int: A tuple representing the decision node or a class label for leaf nodes.
        """
        num_samples, num_features = X.shape
        unique_classes = np.unique(y)

        # Stopping conditions
        if len(unique_classes) == 1 or (self.max_depth is not None and depth >= self.max_depth):
            return unique_classes[0]

        # Find the best split
        best_feature, best_threshold = self._best_split(X, y, num_samples, num_features)

        # Create the tree structure
        if isinstance(best_threshold, list):  # For categorical attributes
            left_indices = np.isin(X[:, best_feature], best_threshold)
            right_indices = ~np.isin(X[:, best_feature], best_threshold)
        else:  # For continuous attributes
            left_indices = X[:, best_feature] < best_threshold
            right_indices = X[:, best_feature] >= best_threshold

        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return (best_feature, best_threshold, left_subtree, right_subtree)

    def _best_split(self, X, y, num_samples, num_features):
        """
        Determines the best feature and threshold to split the data.

        Parameters:
            X (np.ndarray): The feature matrix.
            y (np.ndarray): The target labels.
            num_samples (int): The number of samples.
            num_features (int): The number of features.

        Returns:
            tuple: The best feature index and the best threshold for splitting.
        """
        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature in range(num_features):
            unique_values = np.unique(X[:, feature])

            # Check for categorical or continuous attributes
            if len(unique_values) < 10 and self.binary_tree:  # Binary split for categorical
                for value in unique_values:
                    left_indices = X[:, feature] == value
                    right_indices = X[:, feature] != value

                    gain = self._calculate_gain(y, y[left_indices], y[right_indices])
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_threshold = [value]  # Store as a list for categorical splits
            else:  # Multiple branches for categorical or continuous attributes
                for value in unique_values:
                    left_indices = X[:, feature] == value
                    right_indices = X[:, feature] != value

                    gain = self._calculate_gain(y, y[left_indices], y[right_indices])
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_threshold = [value]

                thresholds, classes = zip(*sorted(zip(X[:, feature], y)))

                for i in range(1, num_samples):
                    if thresholds[i] == thresholds[i - 1]:
                        continue

                    threshold = (thresholds[i] + thresholds[i - 1]) / 2
                    left_classes = classes[:i]
                    right_classes = classes[i:]

                    gain = self._calculate_gain(y, left_classes, right_classes)

                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_threshold = threshold  # Continuous threshold

        return best_feature, best_threshold

    def _calculate_gain(self, parent, left_child, right_child):
        """
        Calculates the gain based on the specified criterion.

        Parameters:
            parent (np.ndarray): The parent labels.
            left_child (np.ndarray): The left child labels.
            right_child (np.ndarray): The right child labels.

        Returns:
            float: The calculated gain.
        """
        if self.criterion == 'information_gain':
            return self._information_gain(parent, left_child, right_child)
        elif self.criterion == 'gain_ratio':
            return self._gain_ratio(parent, left_child, right_child)
        elif self.criterion == 'gini':
            return self._gini_index(parent, left_child, right_child)
        else:
            raise ValueError("Invalid criterion specified.")

    def _information_gain(self, parent, left_child, right_child):
        """
        Calculates the information gain from a split.

        Parameters:
            parent (np.ndarray): The parent labels.
            left_child (np.ndarray): The left child labels.
            right_child (np.ndarray): The right child labels.

        Returns:
            float: The information gain.
        """
        weight_left = len(left_child) / len(parent)
        weight_right = len(right_child) / len(parent)

        return self._entropy(parent) - (weight_left * self._entropy(left_child) + weight_right * self._entropy(right_child))

    def _gain_ratio(self, parent, left_child, right_child):
        """
        Calculates the gain ratio from a split.

        Parameters:
            parent (np.ndarray): The parent labels.
            left_child (np.ndarray): The left child labels.
            right_child (np.ndarray): The right child labels.

        Returns:
            float: The gain ratio.
        """
        gain = self._information_gain(parent, left_child, right_child)
        split_info = self._entropy(left_child) + self._entropy(right_child)
        if split_info == 0:
            return 0
        return gain / split_info

    def _gini_index(self, parent, left_child, right_child):
        """
        Calculates the Gini index for a split.

        Parameters:
            parent (np.ndarray): The parent labels.
            left_child (np.ndarray): The left child labels.
            right_child (np.ndarray): The right child labels.

        Returns:
            float: The Gini index.
        """
        weight_left = len(left_child) / len(parent)
        weight_right = len(right_child) / len(parent)

        return self._gini(parent) - (weight_left * self._gini(left_child) + weight_right * self._gini(right_child))

    def _entropy(self, y):
        """
        Calculates the entropy of a set of labels.

        Parameters:
            y (np.ndarray): The labels.

        Returns:
            float: The calculated entropy.
        """
        unique_classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-9))  # Adding a small value to prevent log(0)

    def _gini(self, y):
        """
        Calculates the Gini impurity of a set of labels.

        Parameters:
            y (np.ndarray): The labels.

        Returns:
            float: The Gini impurity.
        """
        unique_classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def predict(self, X):
        """
        Predicts the class labels for the provided feature matrix.

        Parameters:
            X (np.ndarray): The feature matrix.

        Returns:
            np.ndarray: The predicted class labels.
        """
        return np.array([self._predict_sample(x, self.tree) for x in X])

    def _predict_sample(self, x, tree):
        """
        Recursively predicts the class label for a single sample.

        Parameters:
            x (np.ndarray): A single sample.
            tree (tuple): The decision tree.

        Returns:
            int: The predicted class label.
        """
        if not isinstance(tree, tuple):
            return tree
        
        feature, threshold, left_subtree, right_subtree = tree
        
        if isinstance(threshold, list):  # For categorical splits
            if x[feature] in threshold:
                return self._predict_sample(x, left_subtree)
            else:
                return self._predict_sample(x, right_subtree)
        else:  # For continuous splits
            if x[feature] < threshold:
                return self._predict_sample(x, left_subtree)
            else:
                return self._predict_sample(x, right_subtree)

# Example usage
if __name__ == "__main__":
    # Sample dataset with discrete and continuous attributes
    X = np.array([[2.0, 'A'],
                  [1.0, 'B'],
                  [3.0, 'A'],
                  [5.0, 'B'],
                  [4.0, 'A']])
    
    y = np.array([0, 0, 1, 1, 1])  # Class labels

    # Create and fit the decision tree with Gini index as the criterion and binary tree required
    tree = DecisionTree(max_depth=None, criterion='gain_ratio', binary_tree=True)
    tree.fit(X, y)

    # Make predictions
    predictions = tree.predict(np.array([[2.5, 'A'], [4.0, 'B'], [1.5, 'A']]))
    print(predictions)