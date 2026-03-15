import subprocess
from clearml import Task

PROJECT_NAME = "Sydney Property Recommendation System"
TASK_NAME = "Property Recommendation Pipeline Controller"


def run_command(cmd):
    print(f"\nRunning: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def main():
    task = Task.init(project_name=PROJECT_NAME, task_name=TASK_NAME)

    # Step 1
    run_command("python s1_dataset_artifact.py")

    # Step 2
    run_command("python s2_data_preprocessing.py")

    # Step 3
    run_command("python s3_train_model.py")

    print("\nPipeline completed successfully.")
    task.close()


if __name__ == "__main__":
    main()