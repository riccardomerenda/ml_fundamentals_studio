import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.logistic_regression import LogisticRegression
from utils.preprocessing import StandardScaler
from utils.plotting import plot_learning_curve, plot_decision_boundary
from utils.feature_selection import FeatureSelector, plot_feature_importance
from datasets.data_utils import train_test_split
from utils.metrics import confusion_matrix, precision_recall_f1


def main():
    # Load diabetes dataset
    diabetes = load_diabetes()
    X = diabetes.data
    # Convert to binary classification (median split)
    y = (diabetes.target > np.median(diabetes.target)).astype(int)
    feature_names = diabetes.feature_names
    
    print("Diabetes Dataset:")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Feature names: {feature_names}")
    print(f"Positive class ratio: {np.mean(y):.2f}")
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. Train logistic regression model with all features
    model_all = LogisticRegression(learning_rate=0.1, max_iterations=1000, regularization=0.1)
    model_all.fit(X_train_scaled, y_train)
    
    # Evaluate on test set
    y_pred_all = model_all.predict(X_test_scaled)
    accuracy_all = np.mean(y_pred_all == y_test)
    cm_all = confusion_matrix(y_test, y_pred_all)
    precision_all, recall_all, f1_all = precision_recall_f1(y_test, y_pred_all)
    
    print("\nModel with all features:")
    print(f"Accuracy: {accuracy_all:.4f}")
    print(f"Precision: {precision_all:.4f}")
    print(f"Recall: {recall_all:.4f}")
    print(f"F1 Score: {f1_all:.4f}")
    print("Confusion Matrix:")
    print(cm_all)
    
    # 2. Feature selection
    selector = FeatureSelector()
    
    # Select top 5 features
    selected_features, feature_scores = selector.select_k_best(
        X_train_scaled, y_train, k=5, regression=False)
    
    print("\nSelected top 5 features:")
    for idx in selected_features:
        print(f"  - {feature_names[idx]} (score: {feature_scores[idx]:.4f})")
    
    # Train model with selected features
    X_train_selected = X_train_scaled[:, selected_features]
    X_test_selected = X_test_scaled[:, selected_features]
    
    model_selected = LogisticRegression(learning_rate=0.1, max_iterations=1000, regularization=0.1)
    model_selected.fit(X_train_selected, y_train)
    
    # Evaluate
    y_pred_selected = model_selected.predict(X_test_selected)
    accuracy_selected = np.mean(y_pred_selected == y_test)
    precision_selected, recall_selected, f1_selected = precision_recall_f1(y_test, y_pred_selected)
    
    print("\nModel with selected features:")
    print(f"Accuracy: {accuracy_selected:.4f}")
    print(f"Precision: {precision_selected:.4f}")
    print(f"Recall: {recall_selected:.4f}")
    print(f"F1 Score: {f1_selected:.4f}")
    
    # 3. Visualizations
    
    # Plot learning curves
    fig_learning = plot_learning_curve(model_all, title="Logistic Regression Learning Curve")
    fig_learning.savefig('diabetes_learning_curve.png', dpi=300)
    plt.close(fig_learning)
    
    # Plot feature importance
    fig_importance = plot_feature_importance(
        np.abs(model_all.weights), feature_names, 
        title="Feature Importance for Diabetes Classification")
    fig_importance.savefig('diabetes_feature_importance.png', dpi=300)
    plt.close(fig_importance)
    
    # If we have 2D data after feature selection, plot decision boundary
    if len(selected_features) >= 2:
        # Use the top 2 features for visualization
        top_2_features = selected_features[:2]
        X_train_2d = X_train_scaled[:, top_2_features]
        X_test_2d = X_test_scaled[:, top_2_features]
        
        # Train model on these 2 features
        model_2d = LogisticRegression(learning_rate=0.1, max_iterations=1000, regularization=0.1)
        model_2d.fit(X_train_2d, y_train)
        
        # Plot decision boundary
        fig_decision = plot_decision_boundary(
            model_2d, X_train_2d, y_train, 
            title=f"Decision Boundary ({feature_names[top_2_features[0]]} vs {feature_names[top_2_features[1]]})")
        fig_decision.savefig('diabetes_decision_boundary.png', dpi=300)
        plt.close(fig_decision)
    
    print("\nDiabetes classification completed!")
    print("Generated images:")
    print("- diabetes_learning_curve.png")
    print("- diabetes_feature_importance.png")
    if len(selected_features) >= 2:
        print("- diabetes_decision_boundary.png")


if __name__ == "__main__":
    main()