from pandas import DataFrame
from dataclasses import dataclass
from abc import ABC, abstractmethod

from regularization_methods import RegularizationMethod


class FeatureSelectionStrategy(ABC):
    @abstractmethod
    def process_feature_selection(self,  *args, **kwargs) -> None:
        pass


@dataclass
class EmbeddedFeatureSelection(FeatureSelectionStrategy):
    """Iterative model training to extract features"""
    def process_feature_selection(self, data_frame: DataFrame,
                                  regularization_method: RegularizationMethod) -> None:
        pass

@dataclass
class TreeBasedFeatureSelection(FeatureSelectionStrategy):
    """tree-based ML models"""
    def process_feature_selection(self, data_frame: DataFrame) -> None:
        pass