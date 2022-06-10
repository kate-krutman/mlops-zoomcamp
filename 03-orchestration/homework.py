import pandas as pd
from pendulum import datetime, today, parse
import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner


@task
def get_paths(date):
    if type(date) == str:
        date = parse(date)
    train_dt, val_dt = date.subtract(months=2), date.subtract(months=1)
    y0, m0 = str(train_dt.year), str(train_dt.month).zfill(2)
    y1, m1 = str(val_dt.year), str(val_dt.month).zfill(2)

    main_path = "/Users/katush/data/"
    train_path = main_path + f"fhv_tripdata_{y0}-{m0}.parquet"
    val_path = main_path + f"fhv_tripdata_{y1}-{m1}.parquet"
    return train_path, val_path


@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        print(f"The mean duration of training is {mean_duration}")
    else:
        print(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    print(f"The shape of X_train is {X_train.shape}")
    print(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    print(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    print(f"The MSE of validation is: {mse}")
    return


@flow(task_runner=SequentialTaskRunner())
def main(date=None):
    if date is None:
        date = today().date()

    train_path, val_path = get_paths(date).result()
    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()

    for k, v in {f"model-{date}.pkl": lr, f"dv-{date}.b": dv}.items():
        with open("/Users/katush/Homeworks/mlops-zoomcamp/03-orchestration/"+k, "wb") as f_out:
            pickle.dump(v, f_out)

    run_model(df_val_processed, categorical, dv, lr)

# main("2021-08-15")
