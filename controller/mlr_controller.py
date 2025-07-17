import pandas
from sklearn import linear_model
import logic.multiple_linear_regression as mlr
from sklearn.metrics import mean_squared_error
import time
from utils.generador_csv import multiple_linear_regression_csv


def multiple_linear_regression_data():
    df = pandas.read_csv("data/multiple_linear_regression_data.csv")

    x_df = df[['x1', 'x2']]
    y_df = df['y']
    x = x_df.values.tolist()
    y = y_df.values.tolist()

    return x_df, y_df, x, y


def multiple_linear_regression_sklearn(x, y):
    """
    This function uses the sklearn multiple linear regression module to fit the data and predict the output.
    It returns the slopes, intercept, error and time taken to fit the data.
    """

    start_time = time.time()

    regr = linear_model.LinearRegression()
    regr.fit(x, y)
    y_pred = regr.predict(x)
    error = mean_squared_error(y, y_pred)

    end_time = time.time()
    time_taken = end_time - start_time
    return regr.coef_, regr.intercept_, error, time_taken


def multiple_linear_regression_coursera(x, y):
    """
    This function uses the coursera multiple linear regression module to fit the data and predict the output.
    It returns the slopes, intercept, error and time taken to fit the data.
    """

    start_time = time.time()

    w, b, error = mlr.gradient_descent(x, y, 0.01, 10000)

    end_time = time.time()
    time_taken = end_time - start_time
    return w, b, error, time_taken


def multiple_linear_regression_custom(x, y):
    """
    Again, this function uses all the knowledge of the course Probability and Statistics of Los Andes University.
    It returns the slopes, intercept, error and time taken to fit the data.
    """

    start_time = time.time()

    w, b, error = mlr.multiple_linear_regression(x, y)

    end_time = time.time()
    time_taken = end_time - start_time
    return w, b, error, time_taken


def multiple_linear_regression_func():

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
