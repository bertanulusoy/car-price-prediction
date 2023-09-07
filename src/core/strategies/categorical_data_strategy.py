import logging

from pandas import DataFrame
from abc import ABC, abstractmethod

# A logger for this file
log = logging.getLogger(__name__)


class CategoricalDataStrategy(ABC):
    @abstractmethod
    def process_categorical_data_strategy(self, data_frame: DataFrame) -> DataFrame:
        pass


class DropCategoricalStrategy(CategoricalDataStrategy):
    def process_categorical_data_strategy(self, data_frame: DataFrame) -> DataFrame:
        log.info("DropCategoricalStrategy")
        return data_frame.select_dtypes(exclude=['object'])


class OrdinalEncodingStrategy(CategoricalDataStrategy):
    def process_categorical_data_strategy(self, data_frame: DataFrame) -> DataFrame:
        print("OrdinalEncodingStrategy")
        return DataFrame()  # TO DO: Dataframe


class OneHotEncodingStrategy(CategoricalDataStrategy):
    def process_categorical_data_strategy(self, data_frame: DataFrame) -> DataFrame:
        print("OneHotEncodingStrategy")
        return DataFrame()  # TO DO: Dataframe