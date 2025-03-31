# Machine Learning Fundamentals Studio

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Last Commit](https://img.shields.io/github/last-commit/riccardomerenda/ml_fundamentals_studio)](https://github.com/riccardomerenda/ml_fundamentals_studio/commits/main)
[![Algorithms](https://img.shields.io/badge/algorithms-8-green.svg)](https://github.com/riccardomerenda/ml_fundamentals_studio#project-overview)
[![Open Issues](https://img.shields.io/github/issues/riccardomerenda/ml_fundamentals_studio)](https://github.com/riccardomerenda/ml_fundamentals_studio/issues)
[![Pull Requests](https://img.shields.io/github/issues-pr/riccardomerenda/ml_fundamentals_studio)](https://github.com/riccardomerenda/ml_fundamentals_studio/pulls)
[![Stars](https://img.shields.io/github/stars/riccardomerenda/ml_fundamentals_studio)](https://github.com/riccardomerenda/ml_fundamentals_studio/stargazers)
[![Forks](https://img.shields.io/github/forks/riccardomerenda/ml_fundamentals_studio)](https://github.com/riccardomerenda/ml_fundamentals_studio/network)

# ML Fundamentals Journey

This repository documents my journey learning fundamental machine learning concepts by implementing algorithms from scratch. Rather than just using high-level libraries, I've built core ML algorithms step-by-step to deepen my understanding of the underlying mathematics and principles.

## Project Overview

This project implements and visualizes key machine learning algorithms and concepts:

- **Linear Regression**: Implementation from scratch with gradient descent optimization
- **Feature Engineering**: Scaling techniques and polynomial feature generation
- **Polynomial Regression**: Visualizing the bias-variance tradeoff
- **Regularization**: L2 regularization to combat overfitting
- **Logistic Regression**: Implementation with gradient descent for classification tasks
- **Neural Networks**: Implementation from scratch with forward propagation
- **Decision Trees**: Implementation with information gain and entropy calculations
- **Tree Ensembles**: Random Forests and XGBoost implementations

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
5. **Neural Network Visualization**: Forward propagation visualization and layer activation patterns
6. **Decision Tree Visualization**: Tree structure visualization and split decisions

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
6. Build neural networks from scratch with forward propagation
7. Implement decision trees with information gain
8. Explore tree ensembles (Random Forests and XGBoost)

## Visualizations

The repository includes tools to visualize:
- Gradient descent convergence
- Effects of learning rate on training
- Overfitting vs. underfitting with polynomial features
- Impact of regularization on model complexity
- Decision boundaries for classification problems
- Neural network layer activations and forward propagation
- Decision tree structure and split decisions
- Tree ensemble predictions and feature importance

## Future Additions

As I continue my ML journey, I plan to add:
- Support Vector Machines
- Backpropagation in Neural Networks
- Advanced Neural Network architectures (CNNs, RNNs)
- Clustering algorithms
- Reinforcement Learning basics

## References

This project was inspired by and builds upon knowledge from:
- Andrew Ng's Machine Learning course
- "Hands-On Machine Learning with Scikit-Learn" by Aurélien Géron
- Various online resources and academic papers

---

*This repository serves as both a learning tool and a demonstration of my understanding of core machine learning concepts. All implementations are for educational purposes.*