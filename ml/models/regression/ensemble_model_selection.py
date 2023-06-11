from dataclasses import dataclass
from model_selection import ModelSelection

import matplotlib.pyplot as plt
from numpy import ndarray

from pandas import DataFrame, Series, Index

from sklearn.ensemble import RandomForestRegressor


@dataclass()
class EnsembleModelSelection(ModelSelection):
    # data_frame: DataFrame
    # target: str
    # test_size: float

    n_estimators: int = 100
    max_depth: int = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    n_jobs: int = None
    random_state:int = 42

    __X: ndarray = None
    __y: ndarray = None

    __X_train = None
    __X_test = None
    __y_train = None
    __y_test = None

    __model = None

    def random_forest_model(self):
        self.__model = RandomForestRegressor(n_estimators=self.n_estimators,
                                             max_depth=self.max_depth,
                                             min_samples_split=self.min_samples_split,
                                             min_samples_leaf=self.min_samples_leaf,
                                             n_jobs=self.n_jobs,
                                             random_state=self.random_state)