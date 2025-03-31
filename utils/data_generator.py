import numpy as np

def generate_nonlinear_data(n_samples=1000, noise=0.1, random_state=None):
    """
    Generate a nonlinear classification dataset.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    noise : float
        Standard deviation of Gaussian noise
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    X : array of shape [n_samples, 2]
        The generated samples
    y : array of shape [n_samples]
        The integer labels for classification
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    X = np.random.randn(n_samples, 2)
    y = np.zeros(n_samples)
    
    # Create a nonlinear decision boundary using circles
    y[np.sum(X**2, axis=1) < 2] = 1
    
    # Add some noise
    X += np.random.normal(0, noise, X.shape)
    
    return X, y.astype(int)

def generate_linear_data(n_samples=1000, noise=0.1, random_state=None):
    """
    Generate a linear regression dataset.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    noise : float
        Standard deviation of Gaussian noise
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    X : array of shape [n_samples, 1]
        The generated samples
    y : array of shape [n_samples]
        The target values
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    X = np.random.uniform(-10, 10, (n_samples, 1))
    y = 2 * X.squeeze() + 1
    y += np.random.normal(0, noise, n_samples)
    
    return X, y 