from pandas import DataFrame
from abc import ABC, abstractmethod
from dataclasses import dataclass


class MissingDataStrategy(ABC):
    @abstractmethod
    def process_missing_data_strategy(self, data_frame: DataFrame) -> DataFrame:
        pass


class AssigningNullDataStrategy(MissingDataStrategy):
    """Convert all missing values to null values"""

    def process_missing_data_strategy(self, data_frame: DataFrame) -> DataFrame:
        print("Assigning Missing Values to Null Process Strategy")
        return DataFrame()  # TO DO: Dataframe


class DropMissingDataStrategy(MissingDataStrategy):
    """Dropping all missing values"""
    def process_missing_data_strategy(self, data_frame: DataFrame) -> DataFrame:
        print("Drop Missing Strategy")
        cols_with_missing = [column for column in data_frame.columns
                             if data_frame[column].isnull().any()]
        print(cols_with_missing)
        return DataFrame()  # TO DO: Dataframe


class DataImputationSrategy(MissingDataStrategy):
    def process_missing_data_strategy(self, data_frame: DataFrame) -> DataFrame:
        print("Data Imputation Strategy")
        return DataFrame()  # TO DO: Dataframe


class ExtensionDataImputationStrategy(MissingDataStrategy):
    def process_missing_data_strategy(self, data_frame: DataFrame) -> DataFrame:
        print("Extension Data Imputation Strategy")
        return DataFrame()  # TO DO: Dataframe
