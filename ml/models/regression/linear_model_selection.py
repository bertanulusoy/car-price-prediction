from dataclasses import dataclass
from .model_selection import ModelSelection

import matplotlib.pyplot as plt
from numpy import ndarray

from pandas import DataFrame, Series, Index

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge


@dataclass()
class LinearModelSelection(ModelSelection):
    alpha: int = None
    __kf = None

    def __post_init__(self):
        self.__X = self.data_frame.drop(self.target, axis=1).values
        self.__y = self.data_frame[self.target].values
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(self.__X,
                                                                                        self.__y,
                                                                                        test_size=self.test_size,
                                                                                        random_state=self.random_state)
        self.__kf = KFold(n_splits=6, shuffle=True, random_state=42)

    def linear_model(self):
        self.__model = LinearRegression()
        return self

    def cross_validation_score(self):
        """Cross-validation for R-squared"""
        return cross_val_score(self.__model, self.__X, self.__y, cv=self.__kf)

    def lasso_model(self):
        self.__model = Lasso(alpha=self.alpha)
        return self

    def ridge_model(self):
        self.__model = Ridge(alpha=self.alpha)

    def train(self):
        self.__model.fit(self.__X_train, self.__y_train)
        return self.__X_test, self.__y_test, self.__model

