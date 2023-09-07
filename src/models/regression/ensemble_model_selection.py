from dataclasses import dataclass

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from model_selection import ModelSelection

import matplotlib.pyplot as plt
from numpy import ndarray

from pandas import DataFrame, Series, Index

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor
import xgboost as xgb
from vecstack import stacking


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
    random_state: int = 42

    __X: ndarray = None
    __y: ndarray = None

    __X_train = None
    __X_test = None
    __y_train = None
    __y_test = None

    __model = None
    __stacked_models: list = None
    __stack_train = None
    __stack_test = None

    def __post_init__(self):
        # Create list: stacked_models
        stacked_models = [
            BaggingRegressor(n_estimators=25, random_state=42),
            AdaBoostRegressor(n_estimators=25, random_state=42)
        ]
        # stack the models: stack_train, stack_test
        __stack_train, __stack_test = stacking(stacked_models,
                                               self.__X_train,
                                               self.__y_train,
                                               self.__X_test,
                                               regression=True,
                                               mode='oof_pred_bag',
                                               needs_proba=False,
                                               metric=accuracy_score, n_folds=4,
                                               stratified=True,
                                               shuffle=True, random_state=42, verbose=2)

    def xgboost_regressor(self):
        self.__model = xgb.XGBRegressor(objective='reg:linear',
                                        n_estimators=10,
                                        seed=123)
        return self

    def fit_xgboost_model(self):
        self.__model.fit(self.__stack_train, self.__y_train)
        return self.__X_test, self.__y_test, self.__model

    def random_forest_model(self):
        self.__model = RandomForestRegressor(n_estimators=self.n_estimators,
                                             max_depth=self.max_depth,
                                             min_samples_split=self.min_samples_split,
                                             min_samples_leaf=self.min_samples_leaf,
                                             n_jobs=self.n_jobs,
                                             random_state=self.random_state)
