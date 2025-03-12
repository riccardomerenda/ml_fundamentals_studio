import numpy as np
import pandas as pd


def load_csv(filepath, target_column=None, feature_columns=None):
    """
    Load a CSV file and return features and target if specified.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    target_column : str, optional
        Name of the target column. If None, returns all data.
    feature_columns : list of str, optional
        Names of feature columns. If None, uses all columns except target.
        
    Returns
    -------
    X : numpy.ndarray or pandas.DataFrame
        Feature data.
    y : numpy.ndarray or None
        Target data if target_column is specified, otherwise None.
    """
    # Load data
    data = pd.read_csv(filepath)
    
    # Return all data if no target column specified
    if target_column is None:
        return data, None
    
    # Extract target
    y = data[target_column].values
    
    # Extract features
    if feature_columns is None:
        # Use all columns except target
        feature_columns = [col for col in data.columns if col != target_column]
    
    X = data[feature_columns].values
    
    return X, y


def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split arrays into random train and test subsets.
    
    Parameters
    ----------
    X : array-like
        Features data.
    y : array-like
        Target data.
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.
    random_state : int, optional
        Seed for random number generator.
        
    Returns
    -------
    X_train : array-like
        Training features.
    X_test : array-like
        Testing features.
    y_train : array-like
        Training target.
    y_test : array-like
        Testing target.
    """
    # Set random seed if specified
    if random_state is not None:
        np.random.seed(random_state)
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Get number of samples
    n_samples = len(X)
    
    # Calculate number of test samples
    n_test = int(n_samples * test_size)
    
    # Generate random indices
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    # Split data
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test


def generate_synthetic_data(n_samples=100, n_features=1, noise=0.3, 
                           function_type='linear', random_state=None):
    """
    Generate synthetic data for regression or classification.
    
    Parameters
    ----------
    n_samples : int, default=100
        Number of samples to generate.
    n_features : int, default=1
        Number of features to generate.
    noise : float, default=0.3
        Standard deviation of Gaussian noise added to output.
    function_type : str, default='linear'
        Type of function to generate data: 'linear', 'polynomial', 'sine', or 'classification'.
    random_state : int, optional
        Seed for random number generator.
        
    Returns
    -------
    X : numpy.ndarray
        Feature data of shape (n_samples, n_features).
    y : numpy.ndarray
        Target data of shape (n_samples,).
    """
    # Set random seed if specified
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate features
    X = np.random.rand(n_samples, n_features) * 10 - 5  # Range: -5 to 5
    
    # Generate target based on function type
    if function_type == 'linear':
        # y = w1*x1 + w2*x2 + ... + b + noise
        weights = np.random.randn(n_features)
        bias = np.random.randn()
        y = np.dot(X, weights) + bias
        
    elif function_type == 'polynomial':
        # y = a*x^2 + b*x + c + noise (for 1D case)
        if n_features == 1:
            a, b, c = np.random.randn(3)
            y = a * X[:, 0]**2 + b * X[:, 0] + c
        else:
            # For multiple features, use a mix of linear and quadratic terms
            weights_linear = np.random.randn(n_features)
            weights_quad = np.random.randn(n_features)
            bias = np.random.randn()
            y = np.dot(X, weights_linear) + np.sum(weights_quad * X**2, axis=1) + bias
            
    elif function_type == 'sine':
        # y = a*sin(b*x) + c + noise
        if n_features == 1:
            a, b, c = np.random.randn(3)
            a, b = abs(a), abs(b)  # Make positive for clearer patterns
            y = a * np.sin(b * X[:, 0]) + c
        else:
            # For multiple features, use a weighted sum of sines
            weights = np.random.randn(n_features)
            freqs = np.abs(np.random.randn(n_features))
            bias = np.random.randn()
            y = np.sum(weights * np.sin(freqs * X), axis=1) + bias
            
    elif function_type == 'classification':
        # Binary classification with a linear decision boundary
        weights = np.random.randn(n_features)
        bias = np.random.randn()
        logits = np.dot(X, weights) + bias
        probs = 1 / (1 + np.exp(-logits))  # Sigmoid
        y = (probs > 0.5).astype(int)
        return X, y  # No noise for classification
        
    else:
        raise ValueError(f"Unknown function type: {function_type}")
    
    # Add noise
    if noise > 0:
        y += np.random.randn(n_samples) * noise
    
    return X, y