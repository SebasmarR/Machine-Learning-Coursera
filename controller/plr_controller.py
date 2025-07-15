import logic.polynomial_regression as pr
import time


def polynomial_regression(x, y, degree):
    """
    This function performs polynomial regression on the input data.
    Args:
        x (list or np.ndarray): Input features.
        y (list or np.ndarray): Target values.
        degree (int): Degree of the polynomial.

    Returns:
        np.ndarray: Coefficients of the polynomial regression model.
        time: float: Time taken to perform the regression.
    """

    start_time = time.perf_counter()
    w, b, error = pr.polynomial_regression(x, y, degree)
    end_time = time.perf_counter()
    time_taken = end_time - start_time

    return w, b, error, time_taken
