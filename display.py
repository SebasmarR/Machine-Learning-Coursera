import matplotlib.pyplot as plt
from controller.lr_controller import scipy_linear_regression, coursera_linear_regression, custom_linear_regression
from utils.read_csv import read_csv
from utils.generador_csv import linear_regression_csv, multiple_linear_regression_csv
from controller.mlr_controller import multiple_linear_regression_data, multiple_linear_regression_sklearn, multiple_linear_regression_coursera, multiple_linear_regression_custom
"""
The code of the Coursera course is not used in the final version because it needs to many things like:
    - First of all a learning rate, which is difficult to find an exact value without getting a NaN value.
    - Second, it needs a lot of iterations to get a good value, iterations that we don't know how many are needed to get a good value.
    - Third, it is not as fast as the linear regression function that we created.
Also if you don't input the learning rate exactly as the data needs, it will get a NaN value.
"""


def myfunc(x, slope, intercept):
    return slope * x + intercept


def linear_regression_func():

    linear_regression_csv()
    x, y = read_csv('data/linear_regression.csv')
    w, b, error, time = scipy_linear_regression(x, y)
    # w2, b2, error2, time2 = coursera_linear_regression(x, y)
    w3, b3, error3, time3 = custom_linear_regression(x, y)

    print(f"\nSciPy Linear Regression:")
    print(f"Slope: {w}, Intercept: {b}")
    print(f"Error: {error}, Time taken: {time:.4f} seconds")

    """
    print(f"\nCoursera Linear Regression:")
    print(f"Slope: {w2}, Intercept: {b2}")
    print(f"Error: {error2}, Time taken: {time2:.4f} seconds")
    """

    print(f"\nCustom Linear Regression:")
    print(f"Slope: {w3}, Intercept: {b3}")
    print(f"Error: {error3}, Time taken: {time3:.4f} seconds")

    line1 = list(map(lambda xi: myfunc(xi, w, b), x))
    # line2 = list(map(lambda xi: myfunc(xi, w2, b2), x))
    line3 = list(map(lambda xi: myfunc(xi, w3, b3), x))

    plt.scatter(x, y)
    plt.plot(x, line1, color='green', label='SciPy Linear Regression')
    plt.plot(x, line3, color='blue',
             label='Faster Custom Linear Regression')
    plt.show()


def multiple_linear_regression():

    multiple_linear_regression_csv()
    x_df, y_df, x, y = multiple_linear_regression_data()

    w, b, error, time = multiple_linear_regression_sklearn(x_df, y_df)
    # w2, b2, error2, time2 = multiple_linear_regression_coursera(x, y) - We don't use this because it constantly gets NaN values
    w3, b3, error3, time3 = multiple_linear_regression_custom(x, y)

    print(f"\nSklearn Multiple Linear Regression:")
    print(f"Slopes (w): {w}")
    print(f"Intercept (b): {b}")
    print(f"Error: {error}, Time taken: {time:.4f} seconds")

    # print(f"\nCoursera Multiple Linear Regression:")
    # print(f"Slopes (w): {w2}")
    # print(f"Intercept (b): {b2}")
    # print(f"Error: {error2}, Time taken: {time2:.4f} seconds")

    print(f"\nCustom Multiple Linear Regression:")
    print(f"Slopes (w): {w3}")
    print(f"Intercept (b): {b3}")
    print(f"Error: {error3}, Time taken: {time3:.4f} seconds")


print("Regression Analysis Tool: ")
print("This regression analysis tool allows you to perform linear and multiple linear regression with libraries and with custom codes.")
print("Select the type of regression you want to perform:")
print("1. Linear Regression")
print("2. Multiple Linear Regression")
choice = input("Enter your choice (1 or 2): ")
if choice == "1":
    linear_regression_func()

elif choice == "2":
    multiple_linear_regression()

else:
    print("Invalid choice")
