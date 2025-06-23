import time
from utils.read_csv import read_csv
import logic.linear_regression as lr
from scipy import stats


def scipy_linear_regression(x, y):
    """
    This function uses the SciPy library to perform linear regression.
    """
    start_time = time.time()

    slope, intercept, r, p, std_err = stats.linregress(x, y)
    error = lr.error_function(x, y, slope, intercept)
    print(f"Slope: {slope}, Intercept: {intercept}")
    print(f"Error: {error}")

    end_time = time.time()
    time_taken = end_time - start_time

    return slope, intercept, error, time_taken


def coursera_linear_regression_iterations(x, y):
    """
    This is the implementation of the aprox to linear regression that was used in the coursera course.
    It uses a gradient descent algorithm to find the best slope and intercept for the linear regression.
    It uses a learning rate of 0.001 and a maximum of 10000 iterations.
    """
    start_time = time.time()

    slope, intercept, error = lr.gradient_descent(
        x, y, 0, 0, 0.001, iterations=10000)

    print(f"Custom Slope: {slope}, Custom Intercept: {intercept}")
    print(f"Custom Error: {error}")

    end_time = time.time()
    time_taken = end_time - start_time

    return slope, intercept, error, time_taken


def coursera_linear_regression_automatic(x, y):
    """
    This is the implementation of the aprox to linear regression that was used in the coursera course.
    It uses a gradient descent algorithm to find the best slope and intercept for the linear regression.
    It uses a learning rate of 0.001 and automatic convergence.
    """
    start_time = time.time()

    slope, intercept, error = lr.gradient_descent(
        x, y, 0, 0, 0.001, automatic_convergence=True)

    print(f"Custom Slope: {slope}, Custom Intercept: {intercept}")
    print(f"Custom Error: {error}")

    end_time = time.time()
    time_taken = end_time - start_time

    return slope, intercept, error, time_taken


def custom_linear_regression(x, y):
    """
    This functions uses a custom linear regression module to perform linear regression.
    It is based on the MSE (Mean Squared Error), this formula was taught in the Probability and Statistics course of Los Andes University.
    """

    start_time = time.time()

    slope, intercept, error = lr.linear_regression(x, y)
    print(
        f"Faster Custom Slope: {slope}, Faster Custom Intercept: {intercept}")
    print(f"Faster Custom Error: {error}")

    end_time = time.time()
    time_taken = end_time - start_time

    return slope, intercept, error, time_taken
