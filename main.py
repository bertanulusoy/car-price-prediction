from dataclasses import dataclass

import pandas as pd
import numpy as np
from hydra.utils import instantiate

import logging
from omegaconf import DictConfig
import hydra

from pandas import DataFrame, Series


from sklearn.metrics import mean_squared_error

from src.strategies.missing_data_strategy import MissingDataStrategy
from src.strategies.categorical_data_strategy import CategoricalDataStrategy

from src.models.regression.linear_model_selection import LinearModelSelection
from src.models.regression.ensemble_model_selection import EnsembleModelSelection
from src.metrics.metric_selection import MetricSelection

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
        self.missing_d_strategy = missing_data_strategy

    def run_strategy(self) -> None:
        data_frame: DataFrame = self.car_data.reading().feature_preprocessing()
        df_without_missing: DataFrame = self.missing_d_strategy.process_missing_data_strategy(data_frame)
        df_vectorized_without_missing = self.cat_d_strategy.process_categorical_data_strategy(df_without_missing)

        model_selection = LinearModelSelection(data_frame=df_vectorized_without_missing, target="msrp", test_size=0.3)
        ensemble_model_selection = EnsembleModelSelection(data_frame=df_vectorized_without_missing, target="msrp", test_size=0.3)
        # Linear Regression
        log.info("Linear Regression")
        X_test, y_test, model = model_selection.linear_model().fit_model()
        print(MetricSelection(X_test=X_test, y_test=y_test, model=model)
              .predict()
              .root_mean_squared_error())
        # Lasso Regression
        log.info("Lasso Regression")
        X_test, y_test, model = model_selection.lasso_regularized_model().fit_model()
        print(MetricSelection(X_test=X_test, y_test=y_test, model=model)
              .predict()
              .root_mean_squared_error())
        # Ridge Regression
        log.info("Ridge Regression")
        X_test, y_test, model = model_selection.regularized_ridge_model().fit_model()
        print(MetricSelection(X_test=X_test, y_test=y_test, model=model)
              .predict()
              .root_mean_squared_error())
        # xgboost regressor
        ensemble_model_selection.xgboost_regressor()

        # names = df_vectorized_without_missing.drop(['msrp'], axis=1).columns
        # lasso = Lasso(alpha=0.1)
        # lasso_coef = lasso.fit(X, y).coef_
        # print(lasso_coef)
        # _ = plt.plot(range(len(names)), lasso_coef)
        # _ = plt.xticks(range(len(names)), names, rotation=60)
        # _ = plt.ylabel('Coefficients')
        # plt.show()

        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        lasso = Lasso(alpha=0.1)
        lasso.fit(X_train, y_train)
        lasso_pred = lasso.predict(X_test)
        print(lasso.score(X_test, y_test))
        """

        # reg = LinearRegression()
        # cv_scores = cross_val_score(reg, X, y, cv=10)
        # print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))


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
                      categorical_data_strategy=categorical_data_strategy) \
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
