import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.polynomial_regression import PolynomialRegression
from utils.plotting import plot_regression_results, plot_overfitting_curve
from datasets.data_utils import generate_synthetic_data, train_test_split


def main():
    # Generate synthetic data
    X, y = generate_synthetic_data(
        n_samples=100, 
        n_features=1, 
        noise=0.3, 
        function_type='polynomial', 
        random_state=42
    )
    
    # Reshape X for 1D case
    X = X.flatten()
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train models with different polynomial degrees
    degrees = [1, 2, 3, 10]
    
    plt.figure(figsize=(15, 10))
    for i, degree in enumerate(degrees):
        # Create and fit model
        model = PolynomialRegression(degree=degree, learning_rate=0.01, 
                                     max_iterations=5000, regularization=0.0)
        model.fit(X_train, y_train)
        
        # Plot the results
        plt.subplot(2, 2, i + 1)
        plt.scatter(X_train, y_train, color='blue', label='Training data')
        plt.scatter(X_val, y_val, color='green', label='Validation data')
        
        # Generate points for the model curve
        X_plot = np.linspace(min(X) - 1, max(X) + 1, 100)
        y_plot = model.predict(X_plot)
        
        plt.plot(X_plot, y_plot, color='red', linewidth=2, 
                 label=f'Degree {degree} polynomial')
        
        plt.title(f'Polynomial Regression (Degree {degree})')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('polynomial_regression_degrees.png', dpi=300)
    plt.close()
    
    # Plot training vs validation error for different model complexities
    degrees = list(range(1, 15))  # Try degrees 1 through 14
    
    fig = plot_overfitting_curve(
        PolynomialRegression, degrees, X_train, y_train, X_val, y_val,
        title="Polynomial Regression: Model Complexity vs. Error"
    )
    
    fig.savefig('polynomial_regression_overfitting.png', dpi=300)
    plt.close(fig)
    
    # Demonstrate the effect of regularization
    high_degree = 10  # A degree that's likely to overfit
    
    regularization_values = [0, 0.001, 0.1, 1.0, 10.0]
    plt.figure(figsize=(15, 10))
    
    for i, reg_value in enumerate(regularization_values):
        model = PolynomialRegression(degree=high_degree, learning_rate=0.01, 
                                    max_iterations=5000, regularization=reg_value)
        model.fit(X_train, y_train)
        
        # Plot the results
        plt.subplot(2, 3, i + 1)
        plt.scatter(X_train, y_train, color='blue', alpha=0.7, label='Training data')
        plt.scatter(X_val, y_val, color='green', alpha=0.7, label='Validation data')
        
        # Generate points for the model curve
        X_plot = np.linspace(min(X) - 1, max(X) + 1, 100)
        y_plot = model.predict(X_plot)
        
        plt.plot(X_plot, y_plot, color='red', linewidth=2)
        
        # Calculate and display errors
        train_mse = np.mean((model.predict(X_train) - y_train) ** 2)
        val_mse = np.mean((model.predict(X_val) - y_val) ** 2)
        
        plt.title(f'Regularization λ={reg_value}')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.text(min(X), max(y) - 2, f'Train MSE: {train_mse:.4f}\nVal MSE: {val_mse:.4f}')
    
    plt.tight_layout()
    plt.savefig('polynomial_regression_regularization.png', dpi=300)
    plt.close()
    
    print("Polynomial regression visualization completed!")
    print("Generated images:")
    print("- polynomial_regression_degrees.png")
    print("- polynomial_regression_overfitting.png")
    print("- polynomial_regression_regularization.png")


if __name__ == "__main__":
    main()