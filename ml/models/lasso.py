from dataclasses import dataclass

import matplotlib.pyplot as plt

from pandas import DataFrame, Series, Index
from sklearn.linear_model import Lasso


@dataclass()
class LassoModel:
    X: DataFrame = None
    y: Series = None
    __column_names: Index = None

    def __post_init__(self):
        self.__column_names = self.X.columns

    def __compute_coefficients(self):
        # Instantiate lasso regressor
        lasso = Lasso(alpha=0.4)
        # fit the regressor to the data
        lasso.fit(X=self.X, y=self.y)
        # compute and return coefficients
        return lasso.coef_

    def plot_coefficients(self):
        _ = plt.plot(range(len(self.__column_names)), self.__compute_coefficients())
        _ = plt.xticks(range(len(self.__column_names)), self.__column_names, rotation=60)
        _ = plt.ylabel('Coefficients')
        plt.show()



