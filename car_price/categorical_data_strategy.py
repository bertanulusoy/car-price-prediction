from pandas import DataFrame
from abc import ABC, abstractmethod


class CategoricalDataStrategy(ABC):
    @abstractmethod
    def process_categorical_data_strategy(self, data_frame: DataFrame) -> DataFrame:
        pass


class DropCategoricalStrategy(CategoricalDataStrategy):
    def process_categorical_data_strategy(self, data_frame: DataFrame) -> DataFrame:
        print("DropCategoricalStrategy")
        return DataFrame()  # TO DO: Dataframe


class OrdinalEncodingStrategy(CategoricalDataStrategy):
    def process_categorical_data_strategy(self, data_frame: DataFrame) -> DataFrame:
        print("OrdinalEncodingStrategy")
        return DataFrame()  # TO DO: Dataframe


class OneHotEncodingStrategy(CategoricalDataStrategy):
    def process_categorical_data_strategy(self, data_frame: DataFrame) -> DataFrame:
        print("OneHotEncodingStrategy")
        return DataFrame()  # TO DO: Dataframe