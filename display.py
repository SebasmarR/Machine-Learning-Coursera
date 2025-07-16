import matplotlib.pyplot as plt
from utils.read_csv import read_csv
from utils.generador_csv import *
from controller import *
import numpy as np
"""
We have problems implementing the linear regression function with the Coursera course, but when we normalize the data, it works better.
"""


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


def multiple_linear_regression():

    multiple_linear_regression_csv()
    x_df, y_df, x, y = multiple_linear_regression_data()

    w, b, error, time = multiple_linear_regression_sklearn(x_df, y_df)
    w2, b2, error2, time2 = multiple_linear_regression_coursera(x, y)
    w3, b3, error3, time3 = multiple_linear_regression_custom(x, y)

    print(f"\nSklearn Multiple Linear Regression:")
    print(f"Slopes (w): {w}")
    print(f"Intercept (b): {b}")
    print(f"Error: {error}, Time taken: {time:.4f} seconds")

    print(f"\nCoursera Multiple Linear Regression:")
    print(f"Slopes (w): {w2}")
    print(f"Intercept (b): {b2}")
    print(f"Error: {error2}, Time taken: {time2:.4f} seconds")

    print(f"\nCustom Multiple Linear Regression:")
    print(f"Slopes (w): {w3}")
    print(f"Intercept (b): {b3}")
    print(f"Error: {error3}, Time taken: {time3:.4f} seconds")


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


def logistic_regression_func():

    logistic_regression_csv()

    x1, x2, x, y = logistic_regression_data()

    w, b, error, time_taken = custom_coursera_logistic_regression(x, y)
    print(f"Custom Logistic Regression:")
    print(f"Coefficients (w): {w}")
    print(f"Intercept (b): {b}")
    print(f"Error: {error}, Time taken: {time_taken:.4f} seconds")

    w_sklearn, b_sklearn, error_sklearn, time_taken_sklearn = sklearn_logistic_regression(
        x1, x2)
    print(f"\nSklearn Logistic Regression:")
    print(f"Coefficients (w): {w_sklearn}")
    print(f"Intercept (b): {b_sklearn}")
    print(
        f"Error: {error_sklearn}, Time taken: {time_taken_sklearn:.4f} seconds")

    x1_vals = np.array([row[0] for row in x])
    x2_vals = np.array([row[1] for row in x])
    y_vals = np.array(y)

    x1_range = np.linspace(min(x1_vals), max(x1_vals), 100)

    _, x2_decision = decision_boundary(w, b, x1_range)
    _, x2_decision_sklearn = decision_boundary(w_sklearn, b_sklearn, x1_range)

    plt.scatter(x1_vals[y_vals == 0], x2_vals[y_vals == 0],
                color='red', label='Clase 0')
    plt.scatter(x1_vals[y_vals == 1], x2_vals[y_vals == 1],
                color='blue', label='Clase 1')

    plt.plot(x1_range, x2_decision, color='green',
             label='Decision boundary (Custom)')
    plt.plot(x1_range, x2_decision_sklearn, color='purple',
             label='Decision boundary (Sklearn)')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Logistic Regression Decision Boundary')
    plt.legend()
    plt.grid(True)
    plt.show()


print("Regression Analysis Tool: ")
print("This regression analysis tool allows you to perform linear and multiple linear regression with libraries and with custom codes.")
print("Select the type of regression you want to perform:")
print("1. Linear Regression")
print("2. Multiple Linear Regression")
print("3. Polynomial Regression")
print("4. Logistic Regression")
choice = input("Enter your choice: ")
if choice == "1":
    linear_regression_func()

elif choice == "2":
    multiple_linear_regression()

elif choice == "3":
    polynomial_regression_func()

elif choice == "4":
    logistic_regression_func()

else:
    print("Invalid choice")
