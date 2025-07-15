from . import multiple_linear_regression as mlr
import numpy as np


def degrees(x, d):
    """
    Generate polynomial features up to degree d for the input x.
    Args:
        x (list or np.ndarray): Input features.
        d (int): Degree of the polynomial.

    Returns:
        np.ndarray: Polynomial features of the input x.
    """

    x = np.array(x)
    x = np.array(x).reshape(-1, 1)
    t = np.hstack([x**i for i in range(d, 0, -1)])
    return t


def polynomial_regression(x, y, degree):
    """
    Perform polynomial regression on the input data.
    Args:
        x (list or np.ndarray): Input features.
        y (list or np.ndarray): Target values.
        degree (int): Degree of the polynomial.

    Returns:
        np.ndarray: Coefficients of the polynomial regression model.

    """

    X = degrees(x, degree)
    w, b, error = mlr.multiple_linear_regression(X, y)
    return w, b, error
