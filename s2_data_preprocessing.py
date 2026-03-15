import os
import argparse
import joblib
import pandas as pd
from clearml import Task
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_NAME = "Sydney Property Recommendation System"
TASK_NAME = "Step 2 - Data Preprocessing"


def load_raw_data(dataset_task_id: str = None) -> pd.DataFrame:
    if dataset_task_id:
        dataset_task = Task.get_task(task_id=dataset_task_id)
        raw_df = dataset_task.artifacts["raw_dataset"].get()
        return raw_df

    data_path = os.path.join("work_dataset", "domain_properties.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    return pd.read_csv(data_path)


def remove_price_outliers(df: pd.DataFrame) -> pd.DataFrame:
    q1 = df["price"].quantile(0.25)
    q3 = df["price"].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    df_clean = df[(df["price"] >= lower) & (df["price"] <= upper)].copy()
    return df_clean


def preprocess_data(df: pd.DataFrame):
    df = df.copy()

    # drop rows with missing target
    df = df.dropna(subset=["price"])

    # remove outliers
    df_clean = remove_price_outliers(df)

    y = df_clean["price"]
    X = df_clean.drop("price", axis=1).copy()

    # handle date
    if "date_sold" in X.columns:
        X["date_sold"] = pd.to_datetime(X["date_sold"], dayfirst=True, errors="coerce")
        X["sold_year"] = X["date_sold"].dt.year
        X["sold_month"] = X["date_sold"].dt.month
        X = X.drop("date_sold", axis=1)

    # categorical encoding
    categorical_cols = [col for col in ["suburb", "type"] if col in X.columns]
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # fill missing values after feature engineering
    X = X.fillna(0)

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=13126154
    )

    # future sample
    future_sample_x = X_test.tail(2).copy()
    future_sample_y = y_test.tail(2).copy()

    X_test = X_test.iloc[:-2].copy()
    y_test = y_test.iloc[:-2].copy()

    # scale for linear regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    future_sample_x_scaled = scaler.transform(future_sample_x)

    processed = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "future_sample_x": future_sample_x,
        "future_sample_y": future_sample_y,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "future_sample_x_scaled": future_sample_x_scaled,
        "feature_names": X.columns.tolist(),
        "scaler": scaler,
    }

    summary = {
        "train_shape": X_train.shape,
        "test_shape": X_test.shape,
        "future_sample_shape": future_sample_x.shape,
        "feature_count": len(X.columns),
    }

    return processed, summary


def main(dataset_task_id: str = None):
    task = Task.init(project_name=PROJECT_NAME, task_name=TASK_NAME)
    task.connect({"dataset_task_id": dataset_task_id})

    df = load_raw_data(dataset_task_id)
    processed, summary = preprocess_data(df)

    os.makedirs("model_artifacts", exist_ok=True)
    output_path = os.path.join("model_artifacts", "processed_bundle.pkl")
    joblib.dump(processed, output_path)

    task.upload_artifact(name="processed_bundle", artifact_object=output_path)
    task.upload_artifact(name="preprocess_summary", artifact_object=summary)

    print("Preprocessing summary:", summary)
    print(f"Saved processed bundle to: {output_path}")
    print("Step 2 completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_task_id", type=str, default=None)
    args = parser.parse_args()

    main(dataset_task_id=args.dataset_task_id)import os
import argparse
import joblib
import pandas as pd
from clearml import Task
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_NAME = "Sydney Property Recommendation System"
TASK_NAME = "Step 2 - Data Preprocessing"


def load_raw_data(dataset_task_id: str = None) -> pd.DataFrame:
    if dataset_task_id:
        dataset_task = Task.get_task(task_id=dataset_task_id)
        raw_df = dataset_task.artifacts["raw_dataset"].get()
        return raw_df

    data_path = os.path.join("work_dataset", "domain_properties.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    return pd.read_csv(data_path)


def remove_price_outliers(df: pd.DataFrame) -> pd.DataFrame:
    q1 = df["price"].quantile(0.25)
    q3 = df["price"].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    df_clean = df[(df["price"] >= lower) & (df["price"] <= upper)].copy()
    return df_clean


def preprocess_data(df: pd.DataFrame):
    df = df.copy()

    # drop rows with missing target
    df = df.dropna(subset=["price"])

    # remove outliers
    df_clean = remove_price_outliers(df)

    y = df_clean["price"]
    X = df_clean.drop("price", axis=1).copy()

    # handle date
    if "date_sold" in X.columns:
        X["date_sold"] = pd.to_datetime(X["date_sold"], dayfirst=True, errors="coerce")
        X["sold_year"] = X["date_sold"].dt.year
        X["sold_month"] = X["date_sold"].dt.month
        X = X.drop("date_sold", axis=1)

    # categorical encoding
    categorical_cols = [col for col in ["suburb", "type"] if col in X.columns]
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # fill missing values after feature engineering
    X = X.fillna(0)

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=13126154
    )

    # future sample
    future_sample_x = X_test.tail(2).copy()
    future_sample_y = y_test.tail(2).copy()

    X_test = X_test.iloc[:-2].copy()
    y_test = y_test.iloc[:-2].copy()

    # scale for linear regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    future_sample_x_scaled = scaler.transform(future_sample_x)

    processed = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "future_sample_x": future_sample_x,
        "future_sample_y": future_sample_y,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "future_sample_x_scaled": future_sample_x_scaled,
        "feature_names": X.columns.tolist(),
        "scaler": scaler,
    }

    summary = {
        "train_shape": X_train.shape,
        "test_shape": X_test.shape,
        "future_sample_shape": future_sample_x.shape,
        "feature_count": len(X.columns),
    }

    return processed, summary


def main(dataset_task_id: str = None):
    task = Task.init(project_name=PROJECT_NAME, task_name=TASK_NAME)
    task.connect({"dataset_task_id": dataset_task_id})

    df = load_raw_data(dataset_task_id)
    processed, summary = preprocess_data(df)

    os.makedirs("model_artifacts", exist_ok=True)
    output_path = os.path.join("model_artifacts", "processed_bundle.pkl")
    joblib.dump(processed, output_path)

    task.upload_artifact(name="processed_bundle", artifact_object=output_path)
    task.upload_artifact(name="preprocess_summary", artifact_object=summary)

    print("Preprocessing summary:", summary)
    print(f"Saved processed bundle to: {output_path}")
    print("Step 2 completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_task_id", type=str, default=None)
    args = parser.parse_args()

    main(dataset_task_id=args.dataset_task_id)