# controller/__init__.py
from .lr_controller import coursera_linear_regression_automatic, coursera_linear_regression_iterations, scipy_linear_regression, custom_linear_regression
from .mlr_controller import multiple_linear_regression_data, multiple_linear_regression_sklearn, multiple_linear_regression_coursera, multiple_linear_regression_custom
from .lgr_controller import custom_coursera_logistic_regression, sklearn_logistic_regression, logistic_regression_data, decision_boundary
from .plr_controller import polynomial_regression


VERSION = "1.0.0"
__author__ = "Sebastian"
__email__ = "arsebmar@outlook.com"
__status__ = "Development"
__version__ = VERSION
__maintainer__ = "Sebastian"
__maintainer_email__ = "arsebmar@outlook.com"

__all__ = [
    "coursera_linear_regression_automatic",
    "coursera_linear_regression_iterations",
    "scipy_linear_regression",
    "custom_linear_regression",
    "multiple_linear_regression_data",
    "multiple_linear_regression_sklearn",
    "multiple_linear_regression_coursera",
    "multiple_linear_regression_custom",
    "custom_coursera_logistic_regression",
    "polynomial_regression",
    "sklearn_logistic_regression",
    "logistic_regression_data",
    "decision_boundary"
]
