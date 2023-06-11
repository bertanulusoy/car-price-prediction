from dataclasses import dataclass

from sklearn.metrics import mean_squared_error


@dataclass
class Metrics:
    def __root_mean_squared_error(self, y_test, y_pred):
        return mean_squared_error(y_true=y_test, y_pred=y_pred, squared=False)
