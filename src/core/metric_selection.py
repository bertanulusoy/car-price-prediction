from dataclasses import dataclass
from typing import Any

from numpy import ndarray
from sklearn.metrics import mean_squared_error


@dataclass
class MetricSelection:
    X_test: ndarray
    y_test: ndarray
    model: Any = None
    __y_pred: ndarray = None

    def predict(self):
        self.__y_pred = self.model.predict(self.X_test)
        return self

    def r_square(self):
        return self.model.score(self.X_test, self.y_test)

    def root_mean_squared_error(self):
        return mean_squared_error(y_true=self.y_test, y_pred=self.__y_pred, squared=False)

    def print(self):
        print("R^2: {}".format(self.r_square()))
        print("RMSE: {}".format(self.root_mean_squared_error()))
