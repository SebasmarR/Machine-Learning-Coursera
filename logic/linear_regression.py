import numpy as np


def error_function(x, y, w, b):
    """
    Calculate the error for a linear regression model.
    Args:
        x (list): List of input features.
        y (list): List of target values.
        w (float): First coeficient (slope).
        b (float): Second coeficient (intercept).

    Returns:
        float: The mean squared error between the predicted and actual values.
    """

    if len(x) != len(y):
        raise ValueError("Input lists x and y must have the same length.")

    x = np.array(x)
    y = np.array(y)

    total_error = 0.0

    # Calculates the prediction for each point in the dataset
    prediction = np.dot(x, w) + b

    # Calculates the total error for each point in the dataset
    return np.mean((prediction - y) ** 2)


def derivatives(x, y, w, b):
    """
    Calculate the derivatives of the error function with a input of w and b.
    Args:
        x (list): List of input features.
        y (list): List of target values.
        w (float): First coeficient (slope).
        b (float): Second coeficient (intercept).

    Returns:
        tuple: Derivatives with respect to w and b.
    """

    if len(x) != len(y):
        raise ValueError("Input lists x and y must have the same length.")

    x = np.array(x)
    y = np.array(y)

    # Calculates the prediction for each point in the dataset
    prediction = np.dot(x, w) + b

    # Calculates the derivatives for each point in the dataset
    return np.mean((prediction - y) * x), np.mean(prediction - y)


def normalize_data(x):
    """
    Normalize the input features.
    Args:
        x (list): List of input features.

    Returns:
        np.ndarray: Normalized input features.
    """
    x = np.array(x)
    mean = np.mean(x)
    std_dev = np.std(x)

    normalized_x = (x - mean) / std_dev
    return normalized_x, mean, std_dev


def denormalize_coefficients(w, b, mean_x, std_x):
    w_real = w / std_x
    b_real = b - (w * mean_x / std_x)
    return w_real, b_real


def gradient_descent(x, y, w, b, learning_rate, automatic_convergence=False, iterations=None):
    """
    Perform gradient descent to minimize the error function.
    Args:
        x (list): List of input features.
        y (list): List of target values.
        w (float): Initial value for the slope.
        b (float): Initial value for the intercept.
        learning_rate (float): Learning rate for the gradient descent.
        automatic_convergence (bool): If this is True, the function will automatically determine the number of iterations based on convergence criteria.
        iterations (int): Number of iterations to perform.
    Returns:
        tuple: Updated values of w, b and the final error.
    """
    if automatic_convergence and iterations is not None:
        raise ValueError(
            "If automatic_convergence is True, iterations must be None.")
    if not automatic_convergence and iterations is None:
        raise ValueError(
            "If automatic_convergence is False, iterations must be specified.")

    if len(x) != len(y):
        raise ValueError("Input lists x and y must have the same length.")

    # Normalize the input features
    x, mean, std_dev = normalize_data(x)

    # Convert lists to numpy arrays for efficient calculations
    x = np.array(x)
    y = np.array(y)

    if iterations is not None:

        for _ in range(iterations):
            dw, db = derivatives(x, y, w, b)
            w -= learning_rate * dw
            b -= learning_rate * db

        # Calculates the error after the iterations
        error = error_function(x, y, w, b)

    else:
        iterations = 0
        while True:

            iterations += 1
            last_error = error_function(x, y, w, b)

            dw, db = derivatives(x, y, w, b)
            w -= learning_rate * dw
            b -= learning_rate * db

            error = error_function(x, y, w, b)

            if abs(last_error - error) < 1e-3:  # Convergence criteria
                print(f"Converged after {iterations} iterations.")
                break

    # Denormalize coefficients
    w, b = denormalize_coefficients(w, b, mean, std_dev)
    return w, b, error


def linear_regression(x, y):
    """
    Perform linear regression using a faster method.
    Args:
        x (list): List of input features.
        y (list): List of target values.

    Returns:
        tuple: Slope, intercept, and error.
    """

    if len(x) != len(y):
        raise ValueError("Input lists x and y must have the same length.")

    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)

    num = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(len(x)))
    den = sum((x[i] - x_mean) ** 2 for i in range(len(x)))

    w = num / den
    b = y_mean - w * x_mean
    error = error_function(x, y, w, b)

    return w, b, error
