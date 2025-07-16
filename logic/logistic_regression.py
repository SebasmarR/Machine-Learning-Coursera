from .multiple_linear_regression import multiple_linear_regression
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


def derivatives(x, y, coefficients, intercept):
    """
    Calculate the derivatives of the error function with respect to coefficients and intercept.
    Args:
        x (list): List of input features.
        y (list): List of target values.
        coefficients (np.ndarray): Coefficients of the logistic regression model.
        intercept (float): Intercept of the logistic regression model.

    Returns:
        tuple: Derivatives with respect to coefficients and intercept.
    """
    x = np.array(x)
    y = np.array(y)

    if len(x) != len(y):
        raise ValueError("Input lists x and y must have the same length.")

    z = np.dot(x, coefficients) + intercept
    predictions = sigmoid_function(z)
    error = predictions - y
    dw = np.dot(x.T, error)
    db = np.sum(error)

    return dw, db


def normalize_data(x):
    """
    Normalize each feature in the input dataset.
    Args:
        x (list or np.ndarray): Input features.

    Returns:
        tuple: Normalized features, means, and standard deviations per column.
    """
    x = np.array(x)
    mean = np.mean(x, axis=0)
    std_dev = np.std(x, axis=0)

    normalized_x = (x - mean) / std_dev
    return normalized_x, mean, std_dev


def denormalize_coefficients(w, b, mean_x, std_x):
    """
    Convert coefficients from normalized space back to original feature scale.
    Args:
        w (np.ndarray): Coefficients from normalized data.
        b (float): Intercept from normalized data.
        mean_x (np.ndarray): Mean of each feature.
        std_x (np.ndarray): Standard deviation of each feature.

    Returns:
        tuple: Denormalized coefficients and intercept.
    """
    w_real = w / std_x
    b_real = b - np.sum((w * mean_x) / std_x)
    return w_real, b_real


def gradient_descent(x, y, learning_rate=0.01, iterations=1000):
    """
    Perform gradient descent to optimize the logistic regression model.
    Args:
        x (list or np.ndarray): Input features.
        y (list or np.ndarray): Target values.
        learning_rate (float): Learning rate for gradient descent.
        iterations (int): Number of iterations for gradient descent.

    Returns:
        tuple: Optimized coefficients and intercept.
    """

    x = np.array(x)
    y = np.array(y)

    x_normalized, mean, std_dev = normalize_data(x)

    coefficients, intercept, _ = multiple_linear_regression(x_normalized, y)

    for _ in range(iterations):
        dw, db = derivatives(x_normalized, y, coefficients, intercept)
        coefficients -= learning_rate * dw
        intercept -= learning_rate * db

    coefficients, intercept = denormalize_coefficients(
        coefficients, intercept, mean, std_dev)

    error = error_function(x, y, coefficients, intercept)

    return coefficients, intercept, error
