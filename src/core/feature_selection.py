from dataclasses import dataclass
from typing import Any

from numpy import ndarray
from pandas import DataFrame
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import ElasticNet


@dataclass()
class FeatureImportance:
    data_frame: DataFrame
    target: str

    __X: ndarray = None
    __y: ndarray = None

    __column_names: list = None

    def __post_init__(self):
        self.__X = self.data_frame.drop(self.target, axis=1).values
        self.__y = self.data_frame[self.target].values
        self.column_names = self.data_frame.drop(self.target, axis=1).columns

    def with_lasso(self):
        lasso = Lasso(alpha=0.1)
        lasso.fit(X=self.__X, y=self.__y)
        return lasso.coef_

    def with_random_forest_regressor(self) -> Any:
        random_forest = RandomForestRegressor(max_depth=2,
                                              n_estimators=100,
                                              random_state=42,
                                              oob_score=True)
        random_forest.fit(X=self.__X, y=self.__y)
        return random_forest.feature_importances_

    def with_extra_trees_regressor(self):
        extra_trees = ExtraTreesRegressor()
        extra_trees.fit(X=self.__X, y=self.__y)
        return extra_trees.feature_importances_
    
    def with_elastic_net(self):
        en = ElasticNet(alpha=0.1)
        en.fit(self.__X, self.__y)
        return en.coef_

    def plot_lasso_coefficients(self):
        _ = plt.plot(range(len(self.__column_names)), self.with_lasso())
        _ = plt.xticks(range(len(self.__column_names)), self.__column_names, rotation=60)
        _ = plt.ylabel('Coefficients')
        plt.show()

    def plot_coefficients(self):
        plt.bar(self.__column_names, self.with_lasso())
        plt.xticks(rotation=45)
        plt.show()


