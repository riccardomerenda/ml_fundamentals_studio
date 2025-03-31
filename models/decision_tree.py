import numpy as np

class Node:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        """
        Node class for decision tree
        
        Parameters:
        -----------
        feature_idx : int
            The index of the feature to split on
        threshold : float
            The threshold value to split on
        left : Node
            The left child node
        right : Node
            The right child node
        value : float
            The predicted value for leaf nodes
        """
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        """
        Decision Tree classifier
        
        Parameters:
        -----------
        max_depth : int
            The maximum depth of the tree
        min_samples_split : int
            The minimum number of samples required to split a node
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        """
        Build decision tree classifier
        
        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Training data
        y : array-like, shape = [n_samples]
            Target values
        """
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        """
        Recursively grow decision tree
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Check stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           n_labels == 1:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Find best split
        best_feature, best_threshold = self._best_split(X, y)
        
        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Split data
        left_idxs = X[:, best_feature] < best_threshold
        right_idxs = ~left_idxs
        
        # Grow children
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)

        return Node(best_feature, best_threshold, left, right)

    def _best_split(self, X, y):
        """
        Find the best split using Gini impurity
        """
        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature in range(self.n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._information_gain(y, X[:, feature], threshold)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _information_gain(self, y, X_column, threshold):
        """
        Calculate information gain using Gini impurity
        """
        parent_impurity = self._gini_impurity(y)

        # Generate split
        left_idxs = X_column < threshold
        right_idxs = ~left_idxs
        
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # Calculate weighted avg. impurity of children
        n = len(y)
        n_l, n_r = sum(left_idxs), sum(right_idxs)
        
        if n_l == 0 or n_r == 0:
            return 0
        
        impurity_left = self._gini_impurity(y[left_idxs])
        impurity_right = self._gini_impurity(y[right_idxs])
        weighted_impurity = (n_l * impurity_left + n_r * impurity_right) / n

        # Calculate information gain
        information_gain = parent_impurity - weighted_impurity
        return information_gain

    def _gini_impurity(self, y):
        """
        Calculate Gini impurity
        """
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - sum(probabilities ** 2)

    def _most_common_label(self, y):
        """
        Return most common label in y
        """
        return np.argmax(np.bincount(y))

    def predict(self, X):
        """
        Predict class for X
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        """
        Traverse decision tree
        """
        if node.value is not None:
            return node.value

        if x[node.feature_idx] < node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right) 