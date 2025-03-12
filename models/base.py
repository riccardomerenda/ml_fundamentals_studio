from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Abstract base class for all models in the library.
    
    All models should implement fit, predict, and score methods.
    """
    
    @abstractmethod
    def fit(self, X, y):
        """
        Fit the model to training data.
        
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
        pass
    
    @abstractmethod
    def predict(self, X):
        """
        Predict using the model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
            
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted values.
        """
        pass
    
    @abstractmethod
    def score(self, X, y):
        """
        Return a performance score for the model on given data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True values for X.
            
        Returns
        -------
        score : float
            Performance score.
        """
        pass