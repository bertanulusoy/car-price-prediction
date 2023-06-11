from dataclasses import dataclass

from pandas import DataFrame, Series
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score


@dataclass()
class RidgeModel:
    X: DataFrame = None
    y: Series = None

    def __post_init__(self):
        pass

    def regularize(self):
        ridge = Ridge()




