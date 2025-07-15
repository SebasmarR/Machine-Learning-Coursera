from . import linear_regression as lr
import numpy as np


def linear_regression(x, y):
    """
    This function implements a custom linear regression using MSE (Mean Squared Error).
    It calculates the slope and intercept of the best fit line.
    """

    slope, intercept, error = lr.linear_regression(x, y)

    return slope, intercept, error


def sigmoid_function(z):

    z = np.array(z)
    return 1 / (1 + np.exp(-z))


def logistic_regression(x, y):
    """
    This function implements a custom logistic regression made with the Coursera course.
    First it uses a linear regression to find the slope and intercept of the values and then applies the sigmoid function.
    Finally it also calculates the decision boundary.

    Args:
        x (list): List of input features.
        y (list): List of target values.  

    Returns:
        tuple: Coefficients (w), intercept (b), sigmoid values (z), and decision boundary points (x1, x2).
    """

    # Step 1: Linear Regression
    w, b, _ = linear_regression(x, y)

    x = np.array(x)
    w = np.array(w)

    # Step 2: Apply Sigmoid Function
    z = sigmoid_function(np.dot(x, w) + b)

    return w, b, z
