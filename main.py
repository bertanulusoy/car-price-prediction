from dataclasses import dataclass
import pandas as pd
import numpy as np
import click
from hydra.utils import instantiate

import wandb

import logging
from omegaconf import DictConfig, OmegaConf
import hydra

from pandas import DataFrame, Series
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from car_price.missing_data_strategy import MissingDataStrategy, \
    AssigningNullDataStrategy, DropMissingDataStrategy, \
    DataImputationSrategy, ExtensionDataImputationStrategy

from car_price.categorical_data_strategy import CategoricalDataStrategy, \
    DropCategoricalStrategy,  OrdinalEncodingStrategy, OneHotEncodingStrategy

# A logger for this file
log = logging.getLogger(__name__)


@dataclass
class CarData:
    file_path: str
    data_frame: DataFrame = None
    X_train: DataFrame = None
    X_valid: DataFrame = None
    y_train: Series = None
    y_valid: Series = None

    def reading(self):
        missing_values = ["n/a", "na", "--"]  # "transmission type has unknown fields"
        self.data_frame = pd.read_csv(self.file_path, na_values=missing_values)
        return self

    def feature_preprocessing(self) -> DataFrame:
        """
        Make columns names and contents uniform
        (replace all spaces with underscore and lowercase all letters.)
        :return: None
        """
        # making column names uniform
        self.data_frame.columns = self.data_frame \
            .columns.str.lower() \
            .str.replace(' ', '_')
        # making column contents uniform
        string_columns = list(self.data_frame.dtypes[self.data_frame.dtypes == 'object'].index)
        for col in string_columns:
            self.data_frame[col] = self.data_frame[col].str.lower().str.replace(' ', '_')

        return self.data_frame

    def feature_selection(self) -> None:
        """
        Selecting the right features
        :return:
        """
        pass

    def feature_engineering(self) -> None:
        """
        adding new fields or making combinations of the fields
        :return:
        """
        pass

    def model_selection(self):
        pass

    def parameter_optimizer(self):
        pass

    def model_validator(self):
        pass

    def model_score(self):
        """score your model on the test"""
        model = RandomForestRegressor(n_estimators=100, random_state=0)
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_valid)
        return np.sqrt(mean_squared_error(self.y_valid, predictions))


class CarPricePredictor:

    def __init__(self, car_data: CarData,
                 missing_data_strategy: MissingDataStrategy,
                 categorical_data_strategy: CategoricalDataStrategy):
        self.car_data = car_data
        self.cat_d_strategy = categorical_data_strategy
        self.miss_d_strategy = missing_data_strategy

    def run_strategy(self) -> None:
        data_frame: DataFrame = self.car_data.reading().feature_preprocessing()
        df_without_missing: DataFrame = self.miss_d_strategy.process_missing_data_strategy(data_frame)
        df_vectorized_without_missing = self.cat_d_strategy.process_categorical_data_strategy(df_without_missing)
        print(df_vectorized_without_missing)

"""
@click.command()
@click.option("--wandb_project", help="Name of Weights & Biases project")
@click.option("--wandb_entity", help="Name of Weights & Biases entity")
@click.option(
    "--raw_data_path", help="Location where the raw Car Price Prediction data saved."
)
@click.option("--dest_path", help="Location where the resulting files will be saved")
@click.option("--iteration_count", default=5, help="Number of iterations in the sweep")
"""


@hydra.main(version_base="1.3.2", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # print(OmegaConf.to_yaml(cfg, resolve=True))
    log.info("Info level message")
    log.debug("Debug level message")

    # log.info(cfg.wandb)
    car_data = CarData(cfg.wandb.raw_data_path)

    missing_data_strategy = instantiate(cfg.missing_data_strategy)
    categorical_data_strategy = instantiate(cfg.categorical_data_strategy)

    CarPricePredictor(car_data,
                      missing_data_strategy=missing_data_strategy,
                      categorical_data_strategy=DropCategoricalStrategy())\
        .run_strategy()
    """
    sweep_id = wandb.sweep(sweep=OmegaConf.to_object(cfg=cfg.sweep_config),
                           project=cfg.wandb.project,
                           entity=cfg.wandb.entity)
    print(sweep_id)
    """
    # wandb.agent(sweep_id=sweep_id, partial())


if __name__ == "__main__":
    main()

    """
    CarPricePredictor(car,
                      initial_missing_data=InitialMissingData(),
                      missing_data_strategy=DropMissingData(),
                      categorical_data_strategy=DropCategoricalStrategy())\
        .run_strategy()
    """

"""
config_info = {"name": dict(
    id="Bertan",
    title="title",
    description="description"
)}

print(config_info)
"""
