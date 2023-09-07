from dataclasses import dataclass

import numpy as np
from numpy import ndarray
from pandas import DataFrame


from sklearn.linear_model import Lasso, LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


@dataclass()
class Regularization:

    __X_train: ndarray = None
    __y_train: ndarray = None

    def lasso_cv_alpha_(self):
        lasso_cv = LassoCV(alphas=None, cv=10, max_iter=1000)
        lasso_cv.fit(self.__X_train, self.__y_train)
        return lasso_cv.alpha_

    def ridge_cv_alpha_(self):
        ridge_cv = RidgeCV(alphas=np.logspace(-6, 6, num=13))
        ridge_cv.fit(self.__X_train, self.__y_train)
        return ridge_cv.alpha_
