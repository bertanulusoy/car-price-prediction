from dataclasses import dataclass

from numpy import ndarray
from pandas import DataFrame, Series, Index
from sklearn.model_selection import train_test_split


@dataclass()
class ModelSelection:
    data_frame: DataFrame
    target: str
    test_size: float
    random_state: int = 42

    __X: ndarray = None
    __y: ndarray = None

    __X_train = None
    __X_test = None
    __y_train = None
    __y_test = None

    __model = None

    def __post_init__(self):
        self.__X = self.data_frame.drop(self.target, axis=1).values
        self.__y = self.data_frame[self.target].values
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = \
            train_test_split(self.__X,
                             self.__y,
                             test_size=self.test_size,
                             random_state=self.random_state)
