import os
import pandas as pd
from clearml import Task

PROJECT_NAME = "Sydney Property Recommendation System"
TASK_NAME = "Step 1 - Dataset Artifact"


def main():
    task = Task.init(project_name=PROJECT_NAME, task_name=TASK_NAME)

    data_path = os.path.join("work_dataset", "domain_properties.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    print("Dataset loaded successfully.")
    print(df.head())
    print(df.info())
    print(df.isnull().sum())

    summary = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "missing_values": df.isnull().sum().to_dict(),
    }

    task.upload_artifact(name="raw_dataset", artifact_object=df)
    task.upload_artifact(name="dataset_summary", artifact_object=summary)

    print("Uploaded artifacts: raw_dataset, dataset_summary")
    print("Step 1 completed.")


if __name__ == "__main__":
    main()