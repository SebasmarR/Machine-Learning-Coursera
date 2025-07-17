import time
from utils.read_csv import read_csv
import logic.linear_regression as lr
from scipy import stats
from utils.generador_csv import linear_regression_csv
import matplotlib.pyplot as plt


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


def myfunc(x, slope, intercept):
    return slope * x + intercept


def linear_regression_func():

    linear_regression_csv()
    x, y = read_csv('data/linear_regression.csv')
    w, b, error, time = scipy_linear_regression(x, y)
    w2, b2, error2, time2 = coursera_linear_regression_iterations(x, y)
    w3, b3, error3, time3 = custom_linear_regression(x, y)
    w4, b4, error4, time4 = coursera_linear_regression_automatic(x, y)

    print(f"\nSciPy Linear Regression:")
    print(f"Slope: {w}, Intercept: {b}")
    print(f"Error: {error}, Time taken: {time:.4f} seconds")

    print(f"\nCoursera Linear Regression (Iterations):")
    print(f"Slope: {w2}, Intercept: {b2}")
    print(f"Error: {error2}, Time taken: {time2:.4f} seconds")

    print(f"\nCustom Linear Regression:")
    print(f"Slope: {w3}, Intercept: {b3}")
    print(f"Error: {error3}, Time taken: {time3:.4f} seconds")

    print(f"\nCoursera Linear Regression (Automatic Convergence):")
    print(f"Slope: {w4}, Intercept: {b4}")
    print(f"Error: {error4}, Time taken: {time4:.4f} seconds")

    line1 = list(map(lambda xi: myfunc(xi, w, b), x))
    line2 = list(map(lambda xi: myfunc(xi, w2, b2), x))
    line3 = list(map(lambda xi: myfunc(xi, w3, b3), x))
    line4 = list(map(lambda xi: myfunc(xi, w4, b4), x))

    plt.scatter(x, y)
    plt.plot(x, line1, color='green', label='SciPy Linear Regression')
    plt.plot(x, line2, color='red',
             label='Coursera Linear Regression (Iterations)')
    plt.plot(x, line3, color='blue',
             label='Faster Custom Linear Regression')
    plt.plot(x, line4, color='purple',
             label='Coursera Linear Regression (Automatic Convergence)')
    plt.show()
