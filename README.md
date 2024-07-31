## Incremental Machine Learning Model Training and Evaluation
This repository contains a Python script for incrementally training and evaluating machine learning models for vulnerabilities, threats, and anomalies. The script supports saving results in various formats (CSV, Excel, JSON) and provides detailed logging and visualization of model performance.

## Features
Incremental Training: Supports partial fitting to incrementally train models with new data.
Model Evaluation: Evaluates models using accuracy, precision, recall, F1-score, and ROC curves.
Logging: Logs the training and evaluation process with timestamps.
Flexible Output: Saves evaluation results in CSV, Excel, or JSON formats.

## Getting Started
#### Prerequisites
Python 3.x
Required Python packages: numpy, sklearn, joblib, matplotlib, seaborn, openpyxl
Install the required packages using pip:

```sh
pip install numpy scikit-learn joblib matplotlib seaborn openpyxl
```

## Clone the Repository
Clone the repository from GitHub and navigate to the project directory:

```sh
git clone https://github.com/Shellshock9001/Training_Models.git
cd Training_Models
```
## Running the Script
To run the script for the first time and create the initial models:

```sh
python Shellshock9001/Training_Models/training_models.py
```

To run the script periodically to incrementally train and evaluate the models with new data:

```sh
python Shellshock9001/Training_Models/training_models.py
```

## Script Overview
The script training_models.py performs the following steps:

#### Initialization: Loads existing models and scalers if available, or initializes new ones if not.
#### Data Loading: Loads new data from a specified JSON file (data/nmap_data.json).  
#### Incremental Training: Fits the models with new data incrementally.  
#### Model Saving: Saves the updated models and scalers.  

## Evaluation:
Evaluates the models using test data and logs the results, including detailed explanations of each metric.

## Detailed Example Output
The script will print detailed logs of the training and evaluation process. Below is an example of the output you can expect:

```sh
[2024-07-31 10:00:00] Loading new data from data/nmap_data.json...
[2024-07-31 10:00:01] Data loaded successfully.
[2024-07-31 10:00:01] Starting incremental training for vulnerability model...
[2024-07-31 10:00:02] Vulnerability model training completed.
[2024-07-31 10:00:02] Starting incremental training for threat model...
[2024-07-31 10:00:03] Threat model training completed.
[2024-07-31 10:00:03] Starting incremental training for anomaly model...
[2024-07-31 10:00:04] Anomaly model training completed.
[2024-07-31 10:00:04] Saving models and scalers...
[2024-07-31 10:00:05] Models and scalers saved successfully.
[2024-07-31 10:00:05] Evaluating vulnerability model...
[2024-07-31 10:00:06] Model Evaluation Report:
[2024-07-31 10:00:06] Accuracy: 0.85 - The proportion of true results (both true positives and true negatives) among the total number of cases examined.
[2024-07-31 10:00:06] Precision: 0.88 - The ratio of correctly predicted positive observations to the total predicted positives.
[2024-07-31 10:00:06] Recall: 0.87 - The ratio of correctly predicted positive observations to all observations in actual class.
[2024-07-31 10:00:06] F1-Score: 0.87 - The weighted average of Precision and Recall.
[2024-07-31 10:00:06] Classification Report (detailed metrics for each class):
{
"0": {"precision": 0.90, "recall": 0.85, "f1-score": 0.87, "support": 100},
"1": {"precision": 0.85, "recall": 0.90, "f1-score": 0.87, "support": 100}
}
[2024-07-31 10:00:06] Class 0:
[2024-07-31 10:00:06] Precision: 0.90 - The ratio of correctly predicted positive observations to the total predicted positives.
[2024-07-31 10:00:06] Recall: 0.85 - The ratio of correctly predicted positive observations to all observations in actual class.
[2024-07-31 10:00:06] F1-Score: 0.87 - The weighted average of Precision and Recall.
[2024-07-31 10:00:06] Support: 100 - The number of actual occurrences of the class in the provided dataset.
[2024-07-31 10:00:06] Class 1:
[2024-07-31 10:00:06] Precision: 0.85 - The ratio of correctly predicted positive observations to the total predicted positives.
[2024-07-31 10:00:06] Recall: 0.90 - The ratio of correctly predicted positive observations to all observations in actual class.
[2024-07-31 10:00:06] F1-Score: 0.87 - The weighted average of Precision and Recall.
[2024-07-31 10:00:06] Support: 100 - The number of actual occurrences of the class in the provided dataset.
[2024-07-31 10:00:06] Change in accuracy: +0.05 (5.88%)
```
## Output Formats
The script supports saving results in the following formats:

#### CSV: Stores data in a tabular format, each line representing a row and each column separated by commas.
#### Excel: Allows detailed data analysis with multiple sheets, charts, and formatting.
#### JSON: Stores structured data like objects, arrays, and nested fields.

## Automating the Script
You can set up a cron job (on Unix-based systems) to run the script automatically at regular intervals.

#### Example Cron Job to Run the Script Daily at Midnight
Open the cron file:

```sh
crontab -e
```

#### Add the following line to schedule the script to run daily at midnight:

```sh
0 0 * * * /usr/bin/python3 /path/to/your/repository/Shellshock9001/Training_Models/training_models.py
```

#### Replace /path/to/your/repository/Shellshock9001/Training_Models/training_models.py with the actual path to your script.

## Explanation of Metrics
The script evaluates the models and prints/logs the following metrics:

#### Accuracy: The proportion of true results (both true positives and true negatives) among the total number of cases examined.
#### Precision: The ratio of correctly predicted positive observations to the total predicted positives.
#### Recall: The ratio of correctly predicted positive observations to all observations in actual class.
#### F1-Score: The weighted average of Precision and Recall.
#### Support: The number of actual occurrences of the class in the provided dataset.
