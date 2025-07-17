# controller/__init__.py
from .lr_controller import linear_regression_func
from .mlr_controller import multiple_linear_regression_func
from .plr_controller import polynomial_regression_func
from .lgr_controller import logistic_regression_func

VERSION = "1.0.0"
__author__ = "Sebastian"
__email__ = "arsebmar@outlook.com"
__status__ = "Development"
__version__ = VERSION
__maintainer__ = "Sebastian"
__maintainer_email__ = "arsebmar@outlook.com"

__all__ = [
    "linear_regression_func",
    "multiple_linear_regression_func",
    "polynomial_regression_func",
    "logistic_regression_func"
]
