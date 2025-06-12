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

    total_error = 0.0

    # Calculates the total error for each point in the dataset
    for i in range(len(x)):
        prediction = w * x[i] + b
        total_error += (prediction - y[i]) ** 2

    # Calculates the mean squeared error

    mean_squared_error = total_error / len(x)

    return mean_squared_error


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

    dw = 0.0
    db = 0.0

    # Calculates the derivatives for each point in the dataset
    for i in range(len(x)):
        prediction = w * x[i] + b
        dw += (prediction - y[i]) * x[i]
        db += (prediction - y[i])

    return dw / len(x), db / len(x)


def gradient_descent(x, y, w, b, learning_rate, iterations):
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

    for _ in range(iterations):
        dw, db = derivatives(x, y, w, b)
        w -= learning_rate * dw
        b -= learning_rate * db

    # Calculates the error after the iterations
    error = error_function(x, y, w, b)

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
