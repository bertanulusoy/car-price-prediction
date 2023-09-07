import logging

from pandas import DataFrame
from abc import ABC, abstractmethod
from dataclasses import dataclass

# A logger for this file
log = logging.getLogger(__name__)


class MissingDataStrategy(ABC):
    @abstractmethod
    def process_missing_data_strategy(self, data_frame: DataFrame) -> DataFrame:
        pass


class DropMissingDataStrategy(MissingDataStrategy):
    """Dropping all missing values"""
    def process_missing_data_strategy(self, data_frame: DataFrame) -> DataFrame:
        log.info("Drop Missing Strategy")
        if data_frame.isnull().values.any():
            cols_with_missing = [col for col in data_frame.columns
                                 if data_frame[col].isnull().any()]
            # Drop columns with missing values
            return data_frame.drop(cols_with_missing, axis=1)


class ImputeWithZeroDataStrategy(MissingDataStrategy):
    """Convert all missing values with zero values"""

    def process_missing_data_strategy(self, data_frame: DataFrame) -> DataFrame:
        log.info("Assigning Missing Values to Null Process Strategy")
        if data_frame.isnull().values.any():
            return data_frame.fillna(0)


class DataImputationStrategy(MissingDataStrategy):
    def process_missing_data_strategy(self, data_frame: DataFrame) -> DataFrame:
        log.info("Data Imputation Strategy")
        return DataFrame()  # TO DO: Dataframe


class ExtensionDataImputationStrategy(MissingDataStrategy):
    def process_missing_data_strategy(self, data_frame: DataFrame) -> DataFrame:
        log.info("Extension Data Imputation Strategy")
        return DataFrame()  # TO DO: Dataframe
