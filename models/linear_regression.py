import numpy as np
from models.base import BaseModel


class LinearRegression(BaseModel):
    """
    Linear Regression implemented from scratch with gradient descent optimization.
    
    Parameters
    ----------
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
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6, 
                 regularization=0.0, store_history=True):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.regularization = regularization
        self.store_history = store_history
        
        # Attributes that will be set during training
        self.weights = None
        self.bias = None
        self.cost_history = []
        self.weight_history = []
        self.is_fitted = False
        
    def fit(self, X, y):
        """
        Fit the linear regression model using gradient descent.
        
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
        # Convert inputs to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Get dimensions
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        # Clear history if storing it
        if self.store_history:
            self.cost_history = []
            self.weight_history = []
        
        # Gradient descent
        prev_cost = float('inf')
        
        for i in range(self.max_iterations):
            # Forward pass (compute predictions)
            y_pred = self._forward(X)
            
            # Compute cost
            cost = self._compute_cost(X, y, y_pred)
            
            # Store history if needed
            if self.store_history:
                self.cost_history.append(cost)
                self.weight_history.append(np.concatenate([[self.bias], self.weights]))
            
            # Check for convergence
            if abs(prev_cost - cost) < self.tolerance:
                break
                
            prev_cost = cost
            
            # Compute gradients
            dw, db = self._compute_gradients(X, y, y_pred)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """
        Predict using the linear regression model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
            
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted values.
        """
        if not self.is_fitted:
            raise Exception("Model not fitted yet. Call 'fit' with training data first.")
        
        return self._forward(np.array(X))
    
    def _forward(self, X):
        """Compute the predictions for input X."""
        return np.dot(X, self.weights) + self.bias
    
    def _compute_cost(self, X, y, y_pred=None):
        """
        Compute the cost function (Mean Squared Error with optional regularization).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        y_pred : array-like of shape (n_samples,), optional
            Predicted values. If None, they will be computed.
            
        Returns
        -------
        cost : float
            The value of the cost function.
        """
        if y_pred is None:
            y_pred = self._forward(X)
        
        n_samples = len(y)
        
        # Mean Squared Error
        mse = np.mean((y_pred - y) ** 2) / 2
        
        # Add regularization term (exclude bias)
        reg_term = 0
        if self.regularization > 0:
            reg_term = (self.regularization / (2 * n_samples)) * np.sum(self.weights ** 2)
            
        return mse + reg_term
    
    def _compute_gradients(self, X, y, y_pred):
        """
        Compute the gradients of the cost function with respect to weights and bias.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        y_pred : array-like of shape (n_samples,)
            Predicted values.
            
        Returns
        -------
        dw : array-like of shape (n_features,)
            Gradient with respect to weights.
        db : float
            Gradient with respect to bias.
        """
        n_samples = len(y)
        
        # Gradient for weights and bias
        error = y_pred - y
        dw = (1 / n_samples) * np.dot(X.T, error)
        db = np.mean(error)
        
        # Add regularization term (only for weights, not bias)
        if self.regularization > 0:
            dw += (self.regularization / n_samples) * self.weights
            
        return dw, db
    
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
        y_pred = self.predict(X)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - u/v if v != 0 else 0