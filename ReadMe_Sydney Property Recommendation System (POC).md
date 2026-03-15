# Sydney Property Recommendation System (POC)

## Project Overview

This project implements a **Sydney Property Recommendation System**.
 The system processes real estate datasets, performs data preprocessing, trains a recommendation model, and provides housing suggestions based on user preferences.

The project demonstrates a **basic MLOps pipeline using ClearML**, including:

- Dataset tracking
- Data preprocessing
- Model training
- Experiment logging

------

# Team Members

| Name          | Student ID |
| ------------- | ---------- |
| Congtian Su   | 13126154   |
| Zhengyang Luo | 26023960   |
| Shigang Zhang | 25424274   |

------

# Project Structure

```
Sydney-Property-Recommendation-System
│
├── main.py
├── pipeline_from_tasks.py
│
├── s1_dataset_artifact.py
├── s2_data_preprocessing.py
├── s3_train_model.py
│
├── task_hpo.py
│
├── requirements.txt
└── README.md
```

------

# File Description

## main.py

The entry point of the project.

This script runs the pipeline defined in `pipeline_from_tasks.py`.

```
python main.py
```

------

## pipeline_from_tasks.py

This file defines the **ClearML pipeline** and controls the execution order of tasks.

Pipeline stages:

1. Dataset registration
2. Data preprocessing
3. Model training

The pipeline ensures that each stage runs in sequence and logs the process to ClearML.

------

## s1_dataset_artifact.py

Registers the dataset as a **ClearML dataset artifact**.

Functions include:

- Upload dataset
- Create dataset version
- Track dataset usage

This allows reproducibility and dataset management.

------

## s2_data_preprocessing.py

Handles **data preprocessing and feature engineering**.

Typical tasks include:

- Cleaning property data
- Handling missing values
- Feature extraction
- Formatting attributes such as:
  - location
  - property type
  - price
  - number of bedrooms

The processed dataset is then passed to the training stage.

------

## s3_train_model.py

Trains the **property recommendation model**.

The training process includes:

- Loading processed dataset
- Training the recommendation model
- Logging metrics to ClearML
- Saving trained model artifacts

ClearML records:

- experiment parameters
- training metrics
- model versions

------

## task_hpo.py

Optional **Hyperparameter Optimization (HPO)** task.

This script uses **Optuna integrated with ClearML** to search for better model parameters.

Example parameters tuned:

- learning_rate
- weight_decay
- batch_size

This step is optional and is mainly used for **future model improvement**. 

It's not being used yet, but it will be used later when integrating RNNs or improving existing model parameters for training.

------

## requirements.txt

Lists all required Python dependencies for the project.

Install dependencies using:

```
pip install -r requirements.txt
```

------

# Pipeline Workflow

The machine learning workflow follows three main stages:

1. Dataset Management
    Dataset is uploaded and versioned using ClearML.

2. Data Processing
    Raw property data is cleaned and transformed into usable features.

3. Model Training
    A recommendation model is trained and experiment results are logged.

ClearML tracks the entire pipeline, making experiments reproducible.

------

# Running the Project

### Install dependencies

```
pip install -r requirements.txt
```

### Run the full pipeline

```
python main.py
```

The pipeline will execute:

1. Dataset registration
2. Data preprocessing
3. Model training

------

# Technologies Used

- Python
- ClearML
- Pandas
- Scikit-learn
- Optuna (for HPO)
- Machine Learning Pipelines

------

# Dataset

The dataset contains information about **Sydney housing properties**, including:

- property location
- property type
- price range
- number of bedrooms
- other housing attributes

The dataset is tracked using **ClearML Dataset Artifacts** to ensure reproducibility.

------

# Future Improvements

Potential improvements for this project include:

- Hyperparameter optimization using ClearML HPO
- Deep learning recommendation models
- Real-time recommendation API
- Web-based user interface
- Deployment on AWS SageMaker