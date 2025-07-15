import time
import logic.logistic_regression as lgr


def custom_coursera_logistic_regression(x, y):
    """
    This function implements a custom logistic regression using gradient descent.
    It uses a learning rate of 0.01 and a maximum of 10000 iterations.
    """

    start_time = time.time()

    w, b, z = lgr.logistic_regression(x, y)

    end_time = time.time()
    time_taken = end_time - start_time

    return w, b, z, time_taken
