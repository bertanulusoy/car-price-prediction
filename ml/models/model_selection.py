from dataclasses import dataclass

import matplotlib.pyplot as plt
from numpy import ndarray

from pandas import DataFrame, Series, Index

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


@dataclass()
class Model:
    data_frame: DataFrame
    target: str

    __X: ndarray = None
    __y: ndarray = None

    __X_train = None
    __X_test = None
    __y_train = None
    __y_test = None

    __linear_model = None
    # __column_names: Index = None

    def __post_init__(self):
        # self.__column_names = self.data_frame.columns
        self.__X = self.data_frame.drop(self.target, axis=1).values
        self.__y = self.data_frame[self.target].values

    def linear_model(self, test_size):
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(self.__X,
                                                                                        self.__y,
                                                                                        test_size=test_size,
                                                                                        random_state=42)
        self.__linear_model = LinearRegression()

    def train(self):
        self.__linear_model.fit(self.__X_train, self.__y_train)

    def simple_predictions(self):
        return self.__linear_model.predict(self.__X_test)


"""
    def __compute_coefficients(self) -> ndarray:
        self.__linear_model = LinearRegression()
        self.__linear_model.fit(X=self.__X, y=self.__y)
        return self.__linear_model.coef_

    def plot_coefficients(self):
        _ = plt.plot(range(len(self.__column_names)), self.__compute_coefficients())
        _ = plt.xticks(range(len(self.__column_names)), self.__column_names, rotation=60)
        _ = plt.ylabel('Coefficients')
        plt.show()
        
        
    def compute_rsquare(self):
        return self.__linear_model.score(X=self.__X_test, y=self.__y_test)
"""
