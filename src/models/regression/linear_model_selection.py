from dataclasses import dataclass
from .model_selection import ModelSelection

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray

from pandas import DataFrame, Series, Index

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from .regularization import Regularization


@dataclass()
class LinearModelSelection(ModelSelection):
    __kf = None

    def __post_init__(self):
        """
        self.__X = self.data_frame.drop(self.target, axis=1).values
        self.__y = self.data_frame[self.target].values
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = \
            train_test_split(self.__X,
                             self.__y,
                             test_size=self.test_size,
                             random_state=self.random_state)
        """
        self.__kf = KFold(n_splits=6, shuffle=True, random_state=42)

    def linear_model(self):
        self.__model = LinearRegression()
        return self

    def lasso_model(self, alpha: float):
        self.__model = Lasso(alpha=alpha)
        return self

    def lasso_regularized_model(self):
        reg = Regularization(self.__X_train, self.__y_train)
        alpha_ = reg.lasso_cv_alpha_()
        self.lasso_model(alpha=alpha_)
        return self

    def ridge_model(self, alpha: float):
        self.__model = Ridge(alpha=alpha)
        return self

    def regularized_ridge_model(self):
        reg = Regularization(self.__X_train, self.__y_train)
        alpha_ = reg.ridge_cv_alpha_()
        self.ridge_model(alpha=alpha_)
        return self

    def fit_model(self):
        self.__model.fit(self.__X_train, self.__y_train)
        return self.__X_test, self.__y_test, self.__model

    def cross_validation_score(self):
        """Cross-validation for R-squared"""
        cv_score = cross_val_score(self.__model, self.__X, self.__y, cv=self.__kf)
        # cv_score, mean of cv_score, standard deviation of cv_score, 95 % confidence interval of cv_score
        return cv_score, np.mean(cv_score), np.std(cv_score), np.quantile(cv_score, [0.025, 0.975])

