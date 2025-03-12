import numpy as np
from models.linear_regression import LinearRegression


class PolynomialRegression(LinearRegression):
    """
    Polynomial Regression using Linear Regression with polynomial features.
    
    This model transforms the features into polynomial features up to the specified degree,
    then applies linear regression.
    
    Parameters
    ----------
    degree : int, default=2
        The degree of the polynomial features.
    learning_rate : float, default=0.01
        The step size for gradient descent.
    max_iterations : int, default=1000
        Maximum number of iterations for gradient descent.
    tolerance : float, default=1e-6
        Convergence criterion. If the change in cost function is less than 
        tolerance, the algorithm is considered to have converged.
    regularization : float, default=0.0
        L2 regularization parameter (lambda). Set to 0 for no regularization.
    store_history : bool, default=True
        Whether to store the cost and parameter history during training.
    """
    
    def __init__(self, degree=2, learning_rate=0.01, max_iterations=1000, 
                 tolerance=1e-6, regularization=0.0, store_history=True):
        super().__init__(learning_rate, max_iterations, tolerance, 
                         regularization, store_history)
        self.degree = degree
        
    def fit(self, X, y):
        """
        Fit the polynomial regression model.
        
        Transforms the features into polynomial features, then applies 
        linear regression.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        self : returns an instance of self.
        """
        X_poly = self._polynomial_features(X)
        return super().fit(X_poly, y)
    
    def predict(self, X):
        """
        Predict using the polynomial regression model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
            
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted values.
        """
        X_poly = self._polynomial_features(X)
        return super().predict(X_poly)
    
    def score(self, X, y):
        """
        Return the coefficient of determination R^2 of the prediction.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True values for X.
            
        Returns
        -------
        score : float
            R^2 of self.predict(X) with respect to y.
        """
        X_poly = self._polynomial_features(X)
        return super().score(X_poly, y)
    
    def _polynomial_features(self, X):
        """
        Generate polynomial features up to the specified degree.
        
        For example, if X has one feature x and degree=3, the resulting features
        will be [x, x^2, x^3].
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns
        -------
        X_poly : array-like of shape (n_samples, n_polynomial_features)
            The polynomial features.
        """
        X = np.array(X)
        
        # Handle single feature case differently for clarity
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        n_samples, n_features = X.shape
        
        # Initialize with the original features
        X_poly = X.copy()
        
        # Generate polynomial features from degree 2 to self.degree
        for d in range(2, self.degree + 1):
            for i in range(n_features):
                # Add x_i^d feature
                new_feature = X[:, i] ** d
                X_poly = np.column_stack((X_poly, new_feature))
                
                # Optionally, add interaction terms (x_i * x_j, etc.)
                # This can be extended for more complex feature interactions
        
        return X_poly