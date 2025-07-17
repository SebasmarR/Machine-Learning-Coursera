import logic.polynomial_regression as pr
import time
import numpy as np
import matplotlib.pyplot as plt
from utils.generador_csv import polynomial_regression_csv
from utils.read_csv import read_csv


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


def polynomial_regression_func():
    degree = 3
    polynomial_regression_csv(degree)
    x, y = read_csv('data/polynomial_regression_data.csv')
    w, b, error, time_taken = polynomial_regression(x, y, degree)
    print(f"Coefficients (w): {w}")
    print(f"Intercept (b): {b}")
    print(f"Error: {error}, Time taken: {time_taken:.4f} seconds")

    line = [sum(w[i] * (xi ** (degree - i))
                for i in range(len(w))) + b for xi in x]

    x_sorted, line_sorted = zip(*sorted(zip(x, line)))

    plt.scatter(x, y, label='Data Points')
    plt.plot(x_sorted, line_sorted, color='orange', label='Polynomial Fit')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Polynomial Regression')
    plt.legend()
    plt.show()
