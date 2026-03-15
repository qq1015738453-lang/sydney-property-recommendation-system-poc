import os
import argparse
import joblib
import numpy as np
import pandas as pd
from clearml import Task
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

PROJECT_NAME = "Sydney Property Recommendation System"
TASK_NAME = "Step 3 - Model Training"


def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    non_zero_mask = y_true != 0
    return np.mean(
        np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])
    ) * 100


def evaluate(model_name, model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    scores = {
        "Model": model_name,
        "R2": r2_score(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "MAE": mean_absolute_error(y_test, pred),
        "MAPE": mape(y_test, pred),
    }
    return model, scores


def load_processed_bundle(preprocess_task_id: str = None):
    if preprocess_task_id:
        preprocess_task = Task.get_task(task_id=preprocess_task_id)
        artifact = preprocess_task.artifacts["processed_bundle"]
        bundle_path = artifact.get_local_copy()
        return joblib.load(bundle_path)

    local_path = os.path.join("model_artifacts", "processed_bundle.pkl")
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Processed bundle not found: {local_path}")
    return joblib.load(local_path)


def main(preprocess_task_id: str = None):
    task = Task.init(project_name=PROJECT_NAME, task_name=TASK_NAME)
    task.connect({"preprocess_task_id": preprocess_task_id})

    bundle = load_processed_bundle(preprocess_task_id)

    X_train = bundle["X_train"]
    X_test = bundle["X_test"]
    y_train = bundle["y_train"]
    y_test = bundle["y_test"]

    X_train_scaled = bundle["X_train_scaled"]
    X_test_scaled = bundle["X_test_scaled"]

    future_sample_x = bundle["future_sample_x"]
    future_sample_y = bundle["future_sample_y"]
    future_sample_x_scaled = bundle["future_sample_x_scaled"]

    # train models
    lr_model, lr_scores = evaluate(
        "Linear Regression",
        LinearRegression(),
        X_train_scaled, y_train,
        X_test_scaled, y_test
    )

    dt_model, dt_scores = evaluate(
        "Decision Tree",
        DecisionTreeRegressor(
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=13126154
        ),
        X_train, y_train,
        X_test, y_test
    )

    rf_model, rf_scores = evaluate(
        "Random Forest",
        RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_split=10,
            min_samples_leaf=3,
            max_features="sqrt",
            bootstrap=True,
            n_jobs=-1,
            random_state=13126154
        ),
        X_train, y_train,
        X_test, y_test
    )

    results_df = pd.DataFrame([lr_scores, dt_scores, rf_scores]).sort_values(
        by="RMSE"
    ).reset_index(drop=True)

    print("\nModel Comparison:")
    print(results_df)

    best_model_name = results_df.iloc[0]["Model"]
    model_dict = {
        "Linear Regression": lr_model,
        "Decision Tree": dt_model,
        "Random Forest": rf_model,
    }
    best_model = model_dict[best_model_name]

    # future sample predictions
    future_rows = []
    for name, model in model_dict.items():
        if name == "Linear Regression":
            y_pred_future = model.predict(future_sample_x_scaled)
        else:
            y_pred_future = model.predict(future_sample_x)

        for i in range(len(future_sample_x)):
            future_rows.append({
                "Model": name,
                "Sample_Index": int(future_sample_x.iloc[i].name),
                "Predicted_Price": float(y_pred_future[i]),
                "Actual_Price": float(future_sample_y.iloc[i]),
                "Error": float(y_pred_future[i] - future_sample_y.iloc[i]),
            })

    future_predictions_df = pd.DataFrame(future_rows)

    os.makedirs("model_artifacts", exist_ok=True)

    metrics_path = os.path.join("model_artifacts", "metrics.csv")
    future_path = os.path.join("model_artifacts", "future_predictions.csv")
    bundle_path = os.path.join("model_artifacts", "trained_models_bundle.pkl")

    results_df.to_csv(metrics_path, index=False)
    future_predictions_df.to_csv(future_path, index=False)

    model_bundle = {
        "best_model_name": best_model_name,
        "best_model": best_model,
        "all_models": model_dict,
        "metrics": results_df,
        "future_predictions": future_predictions_df,
        "feature_names": bundle["feature_names"],
        "scaler": bundle["scaler"],
    }
    joblib.dump(model_bundle, bundle_path)

    task.upload_artifact(name="metrics_table", artifact_object=metrics_path)
    task.upload_artifact(name="future_predictions", artifact_object=future_path)
    task.upload_artifact(name="trained_models_bundle", artifact_object=bundle_path)

    logger = task.get_logger()
    for _, row in results_df.iterrows():
        logger.report_scalar("R2", row["Model"], value=row["R2"], iteration=0)
        logger.report_scalar("RMSE", row["Model"], value=row["RMSE"], iteration=0)
        logger.report_scalar("MAE", row["Model"], value=row["MAE"], iteration=0)
        logger.report_scalar("MAPE", row["Model"], value=row["MAPE"], iteration=0)

    print(f"\nBest model: {best_model_name}")
    print("Step 3 completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess_task_id", type=str, default=None)
    args = parser.parse_args()

    main(preprocess_task_id=args.preprocess_task_id)