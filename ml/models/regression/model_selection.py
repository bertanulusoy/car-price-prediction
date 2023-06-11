from dataclasses import dataclass

from numpy import ndarray
from pandas import DataFrame, Series, Index


@dataclass()
class ModelSelection:
    data_frame: DataFrame
    target: str
    test_size: float

    __X: ndarray = None
    __y: ndarray = None

    __X_train = None
    __X_test = None
    __y_train = None
    __y_test = None

    __model = None
