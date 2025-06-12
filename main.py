import matplotlib.pyplot as plt
from scipy import stats
import linear_regression as lr
from read_csv import read_csv

# Data with csv
x, y = read_csv('data.csv')

slope, intercept, r, p, std_err = stats.linregress(x, y)


def myfunc(x):
    return slope * x + intercept


print(f"Slope: {slope}, Intercept: {intercept}")
print(f"Error: {lr.error_function(x, y, slope, intercept)}")

"""
This code was used in the coursera course but is not used in the final version because it needs to many things like:
    - First of all a learning rate, which is difficult to find an exact value without getting a NaN value.
    - Second, it needs a lot of iterations to get a good value, iterations that we don't know how many are needed to get a good value.
    - Third, it is not as fast as the linear regression function that we created.

# Using the custom linear regression module
slope2, intercept2, error = lr.gradient_descent(x, y, 0, 0, 0.001, 10000)

print(f"Custom Slope: {slope2}, Custom Intercept: {intercept2}")
print(f"Custom Error: {error}")


def myfunc2(x):
    return slope2 * x + intercept2
"""


# Using the custom linear regressiong (faster method)


slope3, intercept3, error3 = lr.linear_regression(x, y)
print(f"Faster Custom Slope: {slope3}, Faster Custom Intercept: {intercept3}")
print(f"Faster Custom Error: {error3}")


def myfunc3(x):
    return slope3 * x + intercept3


if lr.error_function(x, y, slope3, intercept3) < lr.error_function(x, y, slope, intercept):
    print("Faster custom linear regression has a lower error than SciPy's linear regression.")
    print("The difference in error is: ",
          lr.error_function(x, y, slope, intercept) - lr.error_function(x, y, slope3, intercept3))
else:
    print("SciPy's linear regression has a lower error than the faster custom linear regression.")
    print("The difference in error is: ",
          lr.error_function(x, y, slope3, intercept3) - lr.error_function(x, y, slope, intercept))

mymodel = list(map(myfunc, x))

# mymodel2 = list(map(myfunc2, x))

mymodel3 = list(map(myfunc3, x))

plt.scatter(x, y)
plt.plot(x, mymodel, color='green', label='SciPy Linear Regression')
plt.plot(x, mymodel3, color='blue', label='Faster Custom Linear Regression')
plt.show()
