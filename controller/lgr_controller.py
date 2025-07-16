import time
import logic.logistic_regression as lgr
from sklearn.linear_model import LogisticRegression
from pandas import read_csv


def logistic_regression_data():
    df = read_csv("data/logistic_regression_data.csv")

    x_df = df[['x1', 'x2']]
    y_df = df['y']
    x = x_df.values.tolist()
    y = y_df.values.tolist()

    return x_df, y_df, x, y


def decision_boundary(coefficients, intercept, x1_range):
    """
    Computes the decision boundary line for 2D logistic regression.

    Args:
        coefficients (list or np.ndarray): Coefficients [w1, w2]
        intercept (float): Intercept b
        x1_range (np.ndarray): Range of x1 values to compute x2

    Returns:
        tuple: (x1_range, x2_values) representing the decision boundary line
    """
    return lgr.decision_boundary_line(coefficients, intercept, x1_range)


def custom_coursera_logistic_regression(x, y):
    """
    This function implements a custom logistic regression using gradient descent.
    It uses a learning rate of 0.01 and a maximum of 10000 iterations.
    """

    start_time = time.time()

    w, b, error = lgr.gradient_descent(
        x, y, learning_rate=0.01, iterations=10000)

    end_time = time.time()
    time_taken = end_time - start_time

    return w, b, error, time_taken


def sklearn_logistic_regression(x, y):
    """
    This function uses sklearn's logistic regression to fit the data and predict the output.
    It returns the coefficients, intercept, cross-entropy error, and time taken.
    """
    start_time = time.time()

    model = LogisticRegression()

    model.fit(x, y)
    error = lgr.error_function(x, y, model.coef_[0], model.intercept_[0])

    end_time = time.time()
    time_taken = end_time - start_time

    return model.coef_[0], model.intercept_[0], error, time_taken
