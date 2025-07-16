from . import linear_regression as lr
import numpy as np


def sigmoid_function(z):
    """
    Calculate the sigmoid function for the input z.
    Args:
        z (list or np.ndarray): Input values.
    Returns:
        np.ndarray: Sigmoid function values for the input z.
    """
    z = np.array(z)
    return 1 / (1 + np.exp(-z))


def decision_boundary_line(coefficients, intercept, x1_range):
    """
    Computes the decision boundary line for 2D logistic regression.

    Args:
        coefficients (list or np.ndarray): Coefficients [w1, w2]
        intercept (float): Intercept b
        x1_range (np.ndarray): Range of x1 values to compute x2

    Returns:
        tuple: (x1_range, x2_values) representing the decision boundary line
    """
    w1, w2 = coefficients
    b = intercept

    if w2 == 0:
        raise ValueError(
            "Coefficient w2 cannot be zero for a valid decision boundary.")

    x2 = -(w1 / w2) * x1_range - (b / w2)

    return x1_range, x2


def classify(x, coefficients, intercept):
    """
    Classifies the input features using the logistic regression model.
    Args:
        x (list or np.ndarray): Input features.
        coefficients (np.ndarray): Coefficients of the logistic regression model.
        intercept (float): Intercept of the logistic regression model.

    Returns:
        np.ndarray: Decision boundary values.
    """
    coefficients = np.array(coefficients)

    z = np.dot(x, coefficients) + intercept

    results = np.where(z >= 0, 1, 0)

    return results


def error_function(x, y, coefficients, intercept):
    """
    Calculate the error for a logistic regression model.
    Args:
        x (list): List of input features.
        y (list): List of target values.
        coefficients (np.ndarray): Coefficients of the logistic regression model.
        intercept (float): Intercept of the logistic regression model.

    Returns:
        float: The mean squared error between the predicted and actual values.
    """
    x = np.array(x)
    y = np.array(y)

    if len(x) != len(y):
        raise ValueError("Input lists x and y must have the same length.")

    z = np.dot(x, coefficients) + intercept

    predictions = sigmoid_function(z)

    predictions = np.clip(predictions, 1e-10, 1 - 1e-10)

    total_error = -np.mean(y * np.log(predictions) +
                           (1 - y) * np.log(1 - predictions))

    return total_error
