import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_learning_curve(model, title="Learning Curve"):
    """
    Plot the learning curve of a model.
    
    Parameters
    ----------
    model : object
        The trained model with a cost_history attribute.
    title : str, default="Learning Curve"
        Title for the plot.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    """
    if not hasattr(model, 'cost_history') or not model.cost_history:
        raise ValueError("Model does not have cost history to plot.")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iterations = range(1, len(model.cost_history) + 1)
    ax.plot(iterations, model.cost_history, 'b-')
    
    ax.set_title(title)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.grid(True)
    
    return fig


def plot_regression_results(model, X, y, X_test=None, y_test=None, feature_index=0,
                           title="Regression Results", figsize=(10, 6)):
    """
    Plot regression model predictions against actual data.
    
    If X has more than one feature, uses the specified feature_index for plotting.
    
    Parameters
    ----------
    model : object
        The trained regression model with predict method.
    X : array-like of shape (n_samples, n_features)
        Training features.
    y : array-like of shape (n_samples,)
        Training target values.
    X_test : array-like of shape (n_samples, n_features), optional
        Test features. If provided, test predictions will be plotted as well.
    y_test : array-like of shape (n_samples,), optional
        Test target values.
    feature_index : int, default=0
        Index of the feature to use for the x-axis when X has multiple features.
    title : str, default="Regression Results"
        Title for the plot.
    figsize : tuple, default=(10, 6)
        Figure size.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    """
    X = np.array(X)
    y = np.array(y)
    
    # Convert 1D array to 2D if needed
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    # Ensure feature_index is valid
    if feature_index >= X.shape[1]:
        raise ValueError(f"feature_index {feature_index} is out of bounds for X with shape {X.shape}")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot of training data
    ax.scatter(X[:, feature_index], y, color='blue', alpha=0.7, s=30, label='Training Data')
    
    # If test data provided, plot it
    if X_test is not None and y_test is not None:
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        # Convert 1D array to 2D if needed
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)
        
        ax.scatter(X_test[:, feature_index], y_test, color='green', alpha=0.7, s=30, label='Test Data')
    
    # Plot the model predictions
    if X.shape[1] == 1:  # Simple plot for 1D feature
        X_plot = np.linspace(np.min(X) - 0.5, np.max(X) + 0.5, 100).reshape(-1, 1)
        y_plot = model.predict(X_plot)
        ax.plot(X_plot, y_plot, 'r-', linewidth=2, label='Model')
    else:  # For multi-feature models, we need to duplicate the mean values for plotting
        X_mean = np.mean(X, axis=0)
        X_plot = np.tile(X_mean, (100, 1))
        x_min, x_max = np.min(X[:, feature_index]), np.max(X[:, feature_index])
        x_range = np.linspace(x_min - 0.5, x_max + 0.5, 100)
        X_plot[:, feature_index] = x_range
        y_plot = model.predict(X_plot)
        ax.plot(x_range, y_plot, 'r-', linewidth=2, label='Model')
    
    ax.set_title(title)
    ax.set_xlabel(f'Feature {feature_index}')
    ax.set_ylabel('Target')
    ax.legend()
    ax.grid(True)
    
    return fig


def plot_decision_boundary(X, y, model, title="Decision Boundary"):
    """
    Plot the decision boundary for a 2D dataset
    
    Parameters:
    -----------
    X : array-like, shape = [n_samples, 2]
        Training data
    y : array-like, shape = [n_samples]
        Target values
    model : object
        Trained model with predict method
    title : str
        Title for the plot
    """
    # Set min and max values with some margin
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    # Create a mesh grid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                        np.arange(y_min, y_max, 0.02))
    
    # Make predictions for each point in the mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the contour and training points
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    plt.show()


def plot_overfitting_curve(model_class, degrees, X_train, y_train, X_val, y_val,
                          title="Polynomial Regression Complexity", figsize=(12, 6)):
    """
    Plot training and validation errors for models of different complexity.
    
    Parameters
    ----------
    model_class : class
        The model class to instantiate (e.g., PolynomialRegression).
    degrees : list of int
        List of polynomial degrees to try.
    X_train : array-like of shape (n_samples, n_features)
        Training features.
    y_train : array-like of shape (n_samples,)
        Training target values.
    X_val : array-like of shape (n_samples, n_features)
        Validation features.
    y_val : array-like of shape (n_samples,)
        Validation target values.
    title : str, default="Polynomial Regression Complexity"
        Title for the plot.
    figsize : tuple, default=(12, 6)
        Figure size.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    """
    train_errors = []
    val_errors = []
    
    for degree in degrees:
        # Create and train model
        model = model_class(degree=degree)
        model.fit(X_train, y_train)
        
        # Compute training and validation errors
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        train_mse = np.mean((train_pred - y_train) ** 2)
        val_mse = np.mean((val_pred - y_val) ** 2)
        
        train_errors.append(train_mse)
        val_errors.append(val_mse)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(degrees, train_errors, 'bo-', linewidth=2, label='Training Error')
    ax.plot(degrees, val_errors, 'ro-', linewidth=2, label='Validation Error')
    
    ax.set_xlabel('Polynomial Degree')
    ax.set_ylabel('Mean Squared Error')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    
    return fig


def plot_regularization_path(X, y, lambda_values, model_class, degree=2, figsize=(12, 6)):
    """
    Plot the effect of regularization on model parameters.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Features.
    y : array-like of shape (n_samples,)
        Target values.
    lambda_values : array-like
        Regularization parameter values to try.
    model_class : class
        The model class to instantiate (e.g., LinearRegression, LogisticRegression).
    degree : int, default=2
        Polynomial degree (if using PolynomialRegression).
    figsize : tuple, default=(12, 6)
        Figure size.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    """
    weights = []
    
    for lambda_val in lambda_values:
        if 'Polynomial' in model_class.__name__:
            model = model_class(degree=degree, regularization=lambda_val)
        else:
            model = model_class(regularization=lambda_val)
            
        model.fit(X, y)
        
        # Store weights (excluding bias)
        weights.append(model.weights)
    
    weights = np.array(weights)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    for i in range(weights.shape[1]):
        ax.plot(lambda_values, weights[:, i], '.-', linewidth=2, label=f'Weight {i+1}')
    
    ax.set_xscale('log')
    ax.set_xlabel('Regularization Parameter (λ)')
    ax.set_ylabel('Weight Value')
    ax.set_title('Regularization Path')
    ax.legend()
    ax.grid(True)
    
    return fig


def plot_learning_curve(train_scores, val_scores, title="Learning Curve"):
    """
    Plot learning curves showing training and validation scores
    
    Parameters:
    -----------
    train_scores : array-like
        List of training scores
    val_scores : array-like
        List of validation scores
    title : str
        Title for the plot
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_scores) + 1)
    plt.plot(epochs, train_scores, 'b-', label='Training score')
    plt.plot(epochs, val_scores, 'r-', label='Validation score')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_regression_line(X, y, y_pred, title="Regression Line"):
    """
    Plot regression data points and the fitted line
    
    Parameters:
    -----------
    X : array-like
        Input features
    y : array-like
        True target values
    y_pred : array-like
        Predicted target values
    title : str
        Title for the plot
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Data points')
    plt.plot(X, y_pred, color='red', label='Regression line')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()