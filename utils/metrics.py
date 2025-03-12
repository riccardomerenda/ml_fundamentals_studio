import numpy as np


def mean_squared_error(y_true, y_pred):
    """
    Calculate the mean squared error between true values and predictions.
    
    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
        
    Returns
    -------
    mse : float
        Mean squared error.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)


def r2_score(y_true, y_pred):
    """
    Calculate the coefficient of determination (R^2 score).
    
    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
        
    Returns
    -------
    r2 : float
        R^2 score.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    u = ((y_true - y_pred) ** 2).sum()
    v = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - u/v if v != 0 else 0


def accuracy_score(y_true, y_pred):
    """
    Calculate the accuracy score for classification problems.
    
    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
        
    Returns
    -------
    accuracy : float
        Proportion of correct predictions.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred):
    """
    Calculate the confusion matrix for binary classification.
    
    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
        
    Returns
    -------
    cm : array of shape (2, 2)
        Confusion matrix where cm[0,0] is true negatives,
        cm[0,1] is false positives, cm[1,0] is false negatives,
        and cm[1,1] is true positives.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for i in range(len(y_true)):
        cm[int(y_true[i]), int(y_pred[i])] += 1
    
    return cm


def precision_recall_f1(y_true, y_pred, pos_label=1):
    """
    Calculate precision, recall, and F1 score for binary classification.
    
    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
    pos_label : int, default=1
        The class to report metrics for.
        
    Returns
    -------
    precision : float
        Precision score.
    recall : float
        Recall score.
    f1 : float
        F1 score.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Calculate true positives, false positives, and false negatives
    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
    fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
    
    # Calculate precision, recall, and F1 score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1