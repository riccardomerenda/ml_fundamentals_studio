# ML Fundamentals Journey

This repository documents my journey learning fundamental machine learning concepts by implementing algorithms from scratch. Rather than just using high-level libraries, I've built core ML algorithms step-by-step to deepen my understanding of the underlying mathematics and principles.

## Project Overview

This project implements and visualizes key machine learning algorithms and concepts:

- **Linear Regression**: Implementation from scratch with gradient descent optimization
- **Feature Engineering**: Scaling techniques and polynomial feature generation
- **Polynomial Regression**: Visualizing the bias-variance tradeoff
- **Regularization**: L2 regularization to combat overfitting
- **Logistic Regression**: Implementation with gradient descent for classification tasks

Each implementation includes detailed visualization tools to demonstrate important ML concepts like learning rates, convergence, overfitting/underfitting, and the effects of regularization.

## Repository Structure

- **models/**: Core algorithm implementations
- **utils/**: Helper functions for visualization, metrics calculation, and data preprocessing
- **datasets/**: Sample datasets and data loading utilities
- **notebooks/**: Interactive Jupyter notebooks explaining concepts and implementations
- **examples/**: Example applications of the algorithms on real-world problems

## Key Features

1. **Pure NumPy Implementations**: Algorithms built using only NumPy for transparent understanding
2. **Interactive Visualizations**: Learning curves, decision boundaries, and parameter effects
3. **Comparison with Scikit-learn**: Validating implementations against industry-standard libraries
4. **Comprehensive Documentation**: Detailed explanations of mathematical concepts and implementation details

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Running the Examples

```bash
python examples/housing_price_prediction.py
```

Or explore the notebooks for interactive learning:

```bash
jupyter notebook notebooks/01_linear_regression_basics.ipynb
```

## Learning Path

This repository follows a learning path that mirrors my own ML education:

1. Start with linear regression and gradient descent fundamentals
2. Explore feature engineering and scaling techniques
3. Understand overfitting through polynomial regression
4. Learn regularization methods to improve generalization
5. Implement logistic regression for classification problems

## Visualizations

The repository includes tools to visualize:
- Gradient descent convergence
- Effects of learning rate on training
- Overfitting vs. underfitting with polynomial features
- Impact of regularization on model complexity
- Decision boundaries for classification problems

## Future Additions

As I continue my ML journey, I plan to add:
- Support Vector Machines
- Neural Networks from scratch
- Decision Trees and Random Forests
- Clustering algorithms

## References

This project was inspired by and builds upon knowledge from:
- Andrew Ng's Machine Learning course
- "Hands-On Machine Learning with Scikit-Learn" by Aurélien Géron
- Various online resources and academic papers

---

*This repository serves as both a learning tool and a demonstration of my understanding of core machine learning concepts. All implementations are for educational purposes.*