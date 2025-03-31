import numpy as np
from .decision_tree import DecisionTree

class RandomForest:
    def __init__(self, n_trees=10, max_depth=None, min_samples_split=2):
        """
        Initialize random forest with given parameters.
        
        Parameters:
        -----------
        n_trees : int
            Number of trees in the forest
        max_depth : int or None
            Maximum depth of each tree
        min_samples_split : int
            Minimum number of samples required to split a node
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        
    def _bootstrap_sample(self, X, y):
        """
        Create a bootstrap sample of the data
        
        Parameters:
        -----------
        X : array-like
            Training features
        y : array-like
            Target values
            
        Returns:
        --------
        X_sample : array-like
            Bootstrap sample of features
        y_sample : array-like
            Bootstrap sample of target values
        """
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[idxs], y[idxs]
    
    def fit(self, X, y):
        """
        Train random forest by fitting individual trees
        
        Parameters:
        -----------
        X : array-like
            Training features
        y : array-like
            Target values
        """
        self.trees = []
        
        for _ in range(self.n_trees):
            # Create bootstrap sample
            X_sample, y_sample = self._bootstrap_sample(X, y)
            
            # Create and train tree
            tree = DecisionTree(max_depth=self.max_depth, 
                              min_samples_split=self.min_samples_split)
            tree.fit(X_sample, y_sample)
            
            # Add tree to forest
            self.trees.append(tree)
    
    def predict(self, X):
        """
        Make predictions using majority voting
        
        Parameters:
        -----------
        X : array-like
            Input features
            
        Returns:
        --------
        predictions : array-like
            Predicted class labels
        """
        # Get predictions from all trees
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # Take majority vote
        predictions = np.round(np.mean(tree_predictions, axis=0))
        
        return predictions 