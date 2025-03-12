import numpy as np


class MinMaxScaler:
    """
    Scale features to a given range [0, 1] by default.
    
    This scaler transforms each feature by scaling it to the given range.
    
    Parameters
    ----------
    feature_range : tuple (min, max), default=(0, 1)
        Desired range of transformed data.
    """
    
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.scale_ = None
        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None
        
    def fit(self, X):
        """
        Compute the minimum and maximum to be used for later scaling.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the min and max values.
            
        Returns
        -------
        self : object
            Fitted scaler.
        """
        X = np.array(X)
        
        # Handle 1D arrays
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        
        # Handle zero range case (prevent division by zero)
        self.data_range_[self.data_range_ == 0] = 1
        
        self.min_, self.max_ = self.feature_range
        self.scale_ = (self.max_ - self.min_) / self.data_range_
        
        return self
    
    def transform(self, X):
        """
        Scale features according to feature_range.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data that will be transformed.
            
        Returns
        -------
        Xt : array-like of shape (n_samples, n_features)
            Transformed data.
        """
        if self.min_ is None or self.scale_ is None:
            raise Exception("Scaler not fitted. Call 'fit' first.")
        
        X = np.array(X)
        
        # Handle 1D arrays
        original_shape = X.shape
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Apply transform
        X_scaled = self.min_ + self.scale_ * (X - self.data_min_)
        
        # Restore original shape if needed
        if len(original_shape) == 1:
            X_scaled = X_scaled.ravel()
            
        return X_scaled
    
    def fit_transform(self, X):
        """
        Fit to data, then transform it.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data that will be transformed.
            
        Returns
        -------
        Xt : array-like of shape (n_samples, n_features)
            Transformed data.
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X):
        """
        Undo the scaling of X according to feature_range.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data that will be inverse transformed.
            
        Returns
        -------
        Xt : array-like of shape (n_samples, n_features)
            Inverse transformed data.
        """
        if self.min_ is None or self.scale_ is None:
            raise Exception("Scaler not fitted. Call 'fit' first.")
        
        X = np.array(X)
        
        # Handle 1D arrays
        original_shape = X.shape
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Apply inverse transform
        X_orig = self.data_min_ + (X - self.min_) / self.scale_
        
        # Restore original shape if needed
        if len(original_shape) == 1:
            X_orig = X_orig.ravel()
            
        return X_orig


class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.
    
    The standard score of a sample x is calculated as:
    z = (x - u) / s
    where u is the mean of the training samples and s is the standard deviation.
    """
    
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.var_ = None
        
    def fit(self, X):
        """
        Compute the mean and standard deviation to be used for later scaling.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation.
            
        Returns
        -------
        self : object
            Fitted scaler.
        """
        X = np.array(X)
        
        # Handle 1D arrays
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        self.mean_ = np.mean(X, axis=0)
        self.var_ = np.var(X, axis=0)
        
        # Handle zero variance case (prevent division by zero)
        self.var_[self.var_ == 0] = 1
        self.scale_ = np.sqrt(self.var_)
        
        return self
    
    def transform(self, X):
        """
        Standardize features by removing the mean and scaling to unit variance.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data that will be transformed.
            
        Returns
        -------
        Xt : array-like of shape (n_samples, n_features)
            Transformed data.
        """
        if self.mean_ is None or self.scale_ is None:
            raise Exception("Scaler not fitted. Call 'fit' first.")
        
        X = np.array(X)
        
        # Handle 1D arrays
        original_shape = X.shape
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Apply transform
        X_scaled = (X - self.mean_) / self.scale_
        
        # Restore original shape if needed
        if len(original_shape) == 1:
            X_scaled = X_scaled.ravel()
            
        return X_scaled
    
    def fit_transform(self, X):
        """
        Fit to data, then transform it.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data that will be transformed.
            
        Returns
        -------
        Xt : array-like of shape (n_samples, n_features)
            Transformed data.
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X):
        """
        Undo the standardization of X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data that will be inverse transformed.
            
        Returns
        -------
        Xt : array-like of shape (n_samples, n_features)
            Inverse transformed data.
        """
        if self.mean_ is None or self.scale_ is None:
            raise Exception("Scaler not fitted. Call 'fit' first.")
        
        X = np.array(X)
        
        # Handle 1D arrays
        original_shape = X.shape
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Apply inverse transform
        X_orig = X * self.scale_ + self.mean_
        
        # Restore original shape if needed
        if len(original_shape) == 1:
            X_orig = X_orig.ravel()
            
        return X_orig