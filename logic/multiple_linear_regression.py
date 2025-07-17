import numpy as np


def error_function(x, y, w, b):
    """
    Calculate the error for a linear regression model.
    Args:
        x (list): List of list with the input features.
        y (list): List of target values.
        w (float): List with the slopes.
        b (float): Second coeficient (intercept).

    Returns:
        float: The mean squared error between the predicted and actual values.
    """

    if any(len(xi) != len(w) for xi in x):
        raise ValueError(
            "Cada x[i] debe tener la misma cantidad de variables que w.")

    total_error = 0.0

    # Calculates the total error for each point in the dataset
    for i in range(len(x)):
        prediction = np.dot(x[i], w) + b
        total_error += (prediction - y[i]) ** 2

    # Calculates the mean squeared error

    mean_squared_error = total_error / len(x)

    return mean_squared_error


def derivatives(x, y, w, b):
    """
    Calculate the derivatives of the error function with a input of w and b.
    Args:
        x (list): List of list with the input features.
        y (list): List of target values.
        w (float): List with the slopes.
        b (float): Second coeficient (intercept).

    Returns:
        tuple: Derivatives with respect to w and b.
    """

    if len(x) != len(y):
        raise ValueError("Input lists x and y must have the same length.")

    dw = np.zeros_like(w)
    db = 0.0

    # Calculates the derivatives for each point in the dataset
    for i in range(len(x)):
        prediction = np.dot(x[i], w) + b
        dw += (prediction - y[i]) * np.array(x[i])
        db += (prediction - y[i])

    return dw / len(x), db / len(x)


def normalize_data(x):
    """
    Normalize the input features.
    Args:
        x (list): List of list with the input features.

    Returns:
        np.ndarray: Normalized input features.
    """
    x = np.array(x)
    mean = np.mean(x, axis=0)
    std_dev = np.std(x, axis=0)

    normalized_x = (x - mean) / std_dev
    return normalized_x, mean, std_dev


def denormalize_data(w, b, mean, std_dev):
    """
    Denormalize the input features.
    Args:
        x (np.ndarray): Normalized input features.
        mean (np.ndarray): Mean of the original data.
        std_dev (np.ndarray): Standard deviation of the original data.

    Returns:
        np.ndarray: Normalized input features.
    """

    denormalized_w = w / std_dev
    denormalized_b = b - np.sum((w * mean) / std_dev)

    return denormalized_w, denormalized_b


def gradient_descent(x, y, learning_rate, iterations):
    """
    Perform gradient descent to minimize the error function.
    Args:
        x (list): List of input features.
        y (list): List of target values.
        w (float): Initial value for the slope.
        b (float): Initial value for the intercept.
        learning_rate (float): Learning rate for the gradient descent.
        iterations (int): Number of iterations to perform.
    Returns:
        tuple: Updated values of w, b and the final error.
    """

    # Initialize w as a zero vector of the same length as x[0]
    if len(x) == 0 or len(y) == 0:
        raise ValueError("Input lists x and y must not be empty.")

    x_original = np.array(x)
    x, mean, std_dev = normalize_data(x)
    w = np.zeros(len(x[0]))
    b = 0
    for _ in range(iterations):
        dw, db = derivatives(x, y, w, b)
        w -= learning_rate * dw
        b -= learning_rate * db

    w, b = denormalize_data(w, b, mean, std_dev)
    # Calculates the error after the iterations
    error = error_function(x_original, y, w, b)

    return w, b, error


# _________________________________________________________________________________________________________________________________________________________________________

def multiple_linear_regression(x, y):
    """
    Perform multiple linear regression to find the best fitting line.
    Args:
        x (list): List of input features.
        y (list): List of target values.
    Returns:
        tuple: Coefficients (slopes) and intercept of the best fitting line.
    """
    if len(x) == 0 or len(y) == 0:
        raise ValueError("Input lists x and y must not be empty.")

    if len(x) != len(y):
        raise ValueError("Input lists x and y must have the same length.")

    # Convert x to a numpy array for matrix operations
    x = np.array(x)
    ones = np.ones((len(x), 1))
    x_with_ones = np.hstack((x, ones))

    x_transpose = np.transpose(x_with_ones)

    xTy = np.dot(x_transpose, y)
    xTx = np.dot(x_transpose, x_with_ones)
    xTx_inv = np.linalg.inv(xTx)
    w_full = np.dot(xTx_inv, xTy)

    w = w_full[:-1]
    b = w_full[-1]

    error = error_function(x, y, w, b)

    return w, b, error
