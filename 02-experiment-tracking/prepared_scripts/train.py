import argparse
import os
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import mlflow


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-homework")


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def run(data_path):
    mlflow.sklearn.autolog()

    with mlflow.start_run():
        # mlflow.set_tag("developer", "cristian")

        train_data_path = os.path.join(data_path, "train.pkl")
        valid_data_path = os.path.join(data_path, "valid.pkl")
        X_train, y_train = load_pickle(train_data_path)
        X_valid, y_valid = load_pickle(valid_data_path)

        # mlflow.log_param("train-data-path", train_data_path)
        # mlflow.log_param("valid-data-path", valid_data_path)

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_valid)

        rmse = mean_squared_error(y_valid, y_pred, squared=False)

        # mlflow.log_metric("rmse", rmse)
        # mlflow.log_artifact(local_path="models/rnd_forest_reg.bin", artifact_path="models_pickle")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="./output",
        help="the location where the processed NYC taxi trip data was saved."
    )
    args = parser.parse_args()

    run(args.data_path)
