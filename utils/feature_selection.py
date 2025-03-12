import numpy as np
from models.linear_regression import LinearRegression
from models.logistic_regression import LogisticRegression


class FeatureSelector:
    """
    A class for implementing various feature selection methods.
    
    This class provides methods to select the most relevant features
    for model training based on different techniques.
    
    Methods:
    --------
    select_k_best: Selects top k features based on their correlation with the target
    recursive_feature_elimination: Selects features using recursive feature elimination
    l1_based_selection: Performs feature selection using L1 regularization (Lasso)
    """
    
    def __init__(self):
        pass
    
    def select_k_best(self, X, y, k=5, regression=True):
        """
        Select the k best features based on correlation with target.
        
        For regression, uses absolute correlation coefficient.
        For classification, uses point-biserial correlation coefficient.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The target values.
        k : int, default=5
            Number of top features to select.
        regression : bool, default=True
            Whether this is a regression task. If False, assumes classification.
            
        Returns
        -------
        selected_features : array of shape (k,)
            The indices of the k best features.
        scores : array of shape (n_features,)
            The scores of all features.
        """
        X = np.array(X)
        y = np.array(y)
        
        n_samples, n_features = X.shape
        
        # Calculate correlation between each feature and the target
        scores = np.zeros(n_features)
        for i in range(n_features):
            feature = X[:, i]
            if regression:
                # Pearson correlation coefficient for regression
                corr = np.corrcoef(feature, y)[0, 1]
                scores[i] = abs(corr)  # Use absolute value for ranking
            else:
                # Point-biserial correlation coefficient (simplified) for classification
                # This is equivalent to Pearson correlation when one variable is binary
                corr = np.corrcoef(feature, y)[0, 1]
                scores[i] = abs(corr)
        
        # Handle NaN values (e.g., constant features)
        scores = np.nan_to_num(scores)
        
        # Select the k features with highest scores
        selected_features = np.argsort(scores)[-k:][::-1]  # Sort in descending order
        
        return selected_features, scores
    
    def recursive_feature_elimination(self, X, y, n_features_to_select=5, step=1, regression=True):
        """
        Selects features by recursively eliminating the least important ones.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The target values.
        n_features_to_select : int, default=5
            The number of features to select.
        step : int, default=1
            The number of features to remove at each iteration.
        regression : bool, default=True
            Whether to use linear regression (True) or logistic regression (False).
            
        Returns
        -------
        selected_features : array of shape (n_features_to_select,)
            The indices of the selected features.
        feature_ranks : array of shape (n_features,)
            The ranking of each feature, lower is better.
        """
        X = np.array(X)
        y = np.array(y)
        
        n_samples, n_features = X.shape
        
        # Check if we have enough features
        if n_features_to_select >= n_features:
            return np.arange(n_features), np.ones(n_features)
        
        # Initialize feature masks and ranks
        feature_ranks = np.ones(n_features, dtype=int) * n_features
        remaining_features = np.arange(n_features)
        current_rank = 1
        
        while len(remaining_features) > n_features_to_select:
            # Create a model based on regression or classification task
            if regression:
                model = LinearRegression(learning_rate=0.01, max_iterations=1000, regularization=0.0)
            else:
                model = LogisticRegression(learning_rate=0.01, max_iterations=1000, regularization=0.0)
            
            # Train model on remaining features
            X_current = X[:, remaining_features]
            model.fit(X_current, y)
            
            # For linear regression, use weights magnitude as importance
            # For logistic regression, also use weights magnitude
            feature_importance = np.abs(model.weights)
            
            # Determine how many features to remove in this iteration
            n_to_remove = min(step, len(remaining_features) - n_features_to_select)
            
            # Find the indices of the least important features
            indices_to_remove = np.argsort(feature_importance)[:n_to_remove]
            
            # Map back to original feature indices
            original_indices_to_remove = remaining_features[indices_to_remove]
            
            # Assign current rank to these features
            feature_ranks[original_indices_to_remove] = current_rank
            
            # Update remaining features
            remaining_features = np.delete(remaining_features, indices_to_remove)
            
            # Increment rank
            current_rank += n_to_remove
        
        # The remaining features are the most important ones
        selected_features = remaining_features
        
        return selected_features, feature_ranks
    
    def l1_based_selection(self, X, y, alpha=1.0, threshold=1e-5, regression=True):
        """
        Perform feature selection using L1 regularization (Lasso).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The target values.
        alpha : float, default=1.0
            Regularization strength.
        threshold : float, default=1e-5
            Features with absolute weights below this threshold are considered zero.
        regression : bool, default=True
            Whether this is a regression task (True) or classification (False).
            
        Returns
        -------
        selected_features : array
            The indices of the selected features.
        feature_importance : array
            The importance of each feature (absolute weight value).
        """
        X = np.array(X)
        y = np.array(y)
        
        # Scale features for better convergence
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_scaled = (X - X_mean) / (X_std + 1e-8)
        
        # For L1 regularization, we need to implement a simple Lasso-like algorithm
        # For simplicity, we'll approximate this with a high regularization parameter
        # and feature elimination based on weight magnitude
        if regression:
            model = LinearRegression(learning_rate=0.01, max_iterations=2000, 
                                     regularization=alpha, store_history=True)
        else:
            model = LogisticRegression(learning_rate=0.01, max_iterations=2000, 
                                      regularization=alpha, store_history=True)
        
        # Train the model
        model.fit(X_scaled, y)
        
        # Get feature importance (absolute weight values)
        feature_importance = np.abs(model.weights)
        
        # Select features above threshold
        selected_features = np.where(feature_importance > threshold)[0]
        
        # If no features are selected, take the one with the highest weight
        if len(selected_features) == 0:
            selected_features = np.array([np.argmax(feature_importance)])
        
        return selected_features, feature_importance


def plot_feature_importance(feature_importance, feature_names=None, title="Feature Importance"):
    """
    Plots the importance of features as a bar chart.
    
    Parameters
    ----------
    feature_importance : array-like
        The importance scores for each feature.
    feature_names : array-like, optional
        The names of the features. If None, uses feature indices.
    title : str, default="Feature Importance"
        The title of the plot.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    """
    import matplotlib.pyplot as plt
    
    # Convert to numpy array
    feature_importance = np.array(feature_importance)
    
    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(feature_importance))]
    
    # Sort features by importance
    sorted_idx = np.argsort(feature_importance)
    sorted_importance = feature_importance[sorted_idx]
    sorted_names = [feature_names[i] for i in sorted_idx]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot horizontal bars
    y_pos = np.arange(len(sorted_names))
    ax.barh(y_pos, sorted_importance, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel('Importance')
    ax.set_title(title)
    
    plt.tight_layout()
    return fig