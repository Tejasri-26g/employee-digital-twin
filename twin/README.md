# Employee Productivity Digital Twin

A machine-learning-powered web application that predicts employee
productivity categories and provides an interactive dashboard for
exploring employee attributes and "what-if" scenarios.

> **Project status:** Hackathon prototype. The reported training results
> below are from the training run shared by the team. Confirm that the
> CSV used for the final submission is the assigned hackathon dataset
> before presenting these results as results on that dataset.

## Table of Contents

-   [Overview](#overview)
-   [Problem Statement](#problem-statement)
-   [Objectives](#objectives)
-   [Features](#features)
-   [Workflow](#workflow)
-   [Technology Stack](#technology-stack)
-   [Dataset](#dataset)
-   [Data Preprocessing and Feature
    Engineering](#data-preprocessing-and-feature-engineering)
-   [Machine Learning Models](#machine-learning-models)
-   [Evaluation Results](#evaluation-results)
-   [System Architecture](#system-architecture)
-   [Project Structure](#project-structure)
-   [Run Locally](#run-locally)
-   [Train the Models on a Dataset](#train-the-models-on-a-dataset)
-   [Deployment](#deployment)
-   [Limitations and Future Scope](#limitations-and-future-scope)
-   [References](#references)

## Overview

The Employee Productivity Digital Twin represents an employee's
work-related state using a web dashboard and machine-learning
predictions. It predicts a productivity category---such as High,
Moderate, or Low---from employee attributes. A What-If simulation allows
users to change selected inputs and compare the resulting prediction
with a baseline prediction.

The project demonstrates an end-to-end machine-learning and full-stack
workflow: CSV analysis, preprocessing, feature engineering, model
training and comparison, prediction through a Django backend, and
presentation through a web dashboard.

## Problem Statement

Employee productivity depends on multiple factors, including attendance,
workload, experience, task completion, training, and other work-related
attributes. Reviewing these factors individually may not provide an easy
way to explore possible productivity scenarios.

This project provides a prototype dashboard that combines employee data,
a trained classifier, and interactive scenario predictions. Its outputs
are exploratory model predictions, not proof of employee performance or
of causal relationships between attributes and productivity.

## Objectives

-   Understand and analyze an employee-productivity dataset.
-   Clean and preprocess the data for machine learning.
-   Engineer additional features from existing columns.
-   Train and compare at least three classification algorithms.
-   Evaluate models using accuracy, precision, recall, F1-scores, and
    confusion matrices.
-   Expose model predictions through a Django application.
-   Display employee information, predictions, and model summaries in a
    dashboard.
-   Support What-If simulations by changing inputs and recalculating
    predictions.
-   Deploy the application so it can be accessed through a public web
    link.

## Features

-   Employee productivity prediction.
-   Predicted class and model probability/confidence output.
-   Comparison of K-Nearest Neighbors (KNN), Random Forest, and Decision
    Tree.
-   Dataset summary and exploratory-data-analysis metadata.
-   Feature-importance metadata from the Random Forest model.
-   What-If scenario prediction for changed employee attributes.
-   Django URL routes for the dashboard and API.
-   Saved model artifacts so prediction requests do not retrain the
    models.

The precise screens and available API operations depend on the
implementation in the repository.

## Workflow

1.  **Load CSV:** Read the employee dataset with Pandas.
2.  **Understand the data:** Inspect row and column counts, data types,
    missing values, duplicates, descriptive statistics, class
    distribution, numerical correlations, histograms, and IQR-based
    outlier counts.
3.  **Preprocess:** Remove duplicate rows and records without a target;
    mark invalid numeric values as missing; impute missing values inside
    the ML pipeline.
4.  **Engineer features:** Derive `Task_Efficiency` and `Training_Ratio`
    through the project's feature-engineering module.
5.  **Split the data:** Use an 80/20 stratified train-test split with
    `random_state=42`.
6.  **Train models:** Fit KNN, Random Forest, and Decision Tree
    pipelines.
7.  **Evaluate and compare:** Calculate accuracy, macro precision, macro
    recall, macro F1, weighted F1, and confusion matrices.
8.  **Select a model:** Choose the model with the highest macro
    F1-score.
9.  **Save artifacts:** Save trained pipelines, the selected model,
    processed employee data, model-comparison results, and JSON
    metadata.
10. **Serve predictions:** Django loads saved artifacts and returns
    prediction results through its backend/API.
11. **Display and simulate:** The frontend presents the dashboard and
    lets users compare baseline and modified-input predictions.

## Technology Stack

### Frontend

-   **HTML5** --- page structure (`frontend/index.html`).
-   **CSS3** --- styling (`frontend/style.css`).
-   **JavaScript and React 18** --- interactive dashboard
    (`frontend/app.jsx`).
-   **Babel Standalone** --- browser-side JSX transformation in the
    current HTML setup.
-   **Chart.js** --- charts in the browser.

The current HTML loads React, ReactDOM, Babel, and Chart.js from CDNs.
An internet connection is therefore required for those CDN assets to
load.

### Backend

-   **Python** --- training and prediction logic.
-   **Django** --- web application, URL routing, templates, and API
    integration.
-   **Gunicorn** --- production WSGI server on Render.
-   **WhiteNoise** --- static-file serving, if installed and configured
    as described in the deployment setup.

### Machine Learning and Data Processing

-   **Pandas** --- CSV loading and data analysis.
-   **NumPy** --- numerical operations.
-   **scikit-learn** --- preprocessing, pipelines, classifiers,
    train/test split, and evaluation metrics.
-   **Joblib** --- saving and loading trained models.
-   **JSON** --- metadata for the API and dashboard.

### Storage and Deployment

-   **CSV files** --- dataset and processed employee records.
-   **Pickle/joblib model artifacts (`.pkl`)** --- saved trained models.
-   **SQLite** --- configured as Django's database; the current ML
    employee records are also loaded from a CSV artifact.
-   **Git and GitHub** --- version control and source-code hosting.
-   **Render** --- cloud deployment.
-   **VS Code and Python virtual environment** --- local development.

## Dataset

The training command shared by the team loaded
`06_employee_productivity_digital_twin.csv` and reported **500 rows and
13 columns**.

The code references employee-related attributes such as:

-   Age and experience years
-   Project count
-   Attendance
-   Working hours
-   Tasks completed
-   Task delay count
-   Training hours
-   Meeting hours
-   Performance score
-   Job satisfaction
-   Employee ID
-   Productivity category (target)

The exact column list should be confirmed against the final CSV. The
employee ID is excluded from model inputs, and the target column is
imported from `api/features.py` as `TARGET`.

**Dataset source and permission:** The project run output establishes
the filename and size, but does not establish the dataset's origin,
license, or whether it is synthetic/sample data. Add the assigned
dataset's source, URL/citation, license or permission, and final column
names here before submission. Do not describe the data as collected from
a real organization unless that is verified.

## Data Preprocessing and Feature Engineering

The training script performs the following operations:

-   Reads the CSV with Pandas.
-   Records initial dataset dimensions, data types, missing-value
    counts, duplicate count, and descriptive statistics.
-   Removes duplicate rows and rows with a missing target label.
-   For numeric columns other than the ID column, converts negative
    values to missing values.
-   Converts `Attendance` values above 100 to missing values.
-   Reports IQR-based outlier counts; this calculation reports outliers
    and does not itself remove all outlier rows.
-   Uses median imputation and `StandardScaler` for numerical features.
-   Uses most-frequent imputation and one-hot encoding for categorical
    features.
-   Uses `add_features` from `api/features.py` to derive
    `Task_Efficiency` and `Training_Ratio`.
-   Splits data into 80% training and 20% testing sets using
    stratification and `random_state=42`.

Preprocessing is implemented in scikit-learn pipelines so the
transformations are fitted as part of model training and reused for
prediction.

## Machine Learning Models

Three classifiers are trained:

1.  **K-Nearest Neighbors (KNN):** `n_neighbors=5`. Predicts using the
    labels of nearby training examples. The KNN pipeline also applies
    the project's feature weighting.
2.  **Random Forest:** `n_estimators=200`, `random_state=42`, and
    `class_weight="balanced"`. Combines predictions from an ensemble of
    decision trees.
3.  **Decision Tree:** `max_depth=5`, `random_state=42`, and
    `class_weight="balanced"`. Learns a sequence of feature-based
    decision rules.

The training script selects the model with the highest **macro
F1-score**. It saves each model, the selected model as `best_model.pkl`,
and evaluation metadata.

## Evaluation Results

The following values are copied from the training output shared by the
team:

  ---------------------------------------------------------------------------
  Model          Accuracy        Macro Macro Recall     Macro F1  Weighted F1
                             Precision                           
  ---------- ------------ ------------ ------------ ------------ ------------
  KNN                0.91       0.5792       0.5786       0.5786       0.8996

  Random         **0.93**       0.5885       0.6219       0.6039   **0.9222**
  Forest                                                         

  Decision           0.86   **0.6952**   **0.7078**   **0.7010**       0.8624
  Tree                                                           
  ---------------------------------------------------------------------------

**Selected model: Decision Tree**, because its macro F1-score (0.7010)
is the highest of the three models. Random Forest has the highest
accuracy (0.93), so model selection reflects the chosen macro-F1
objective rather than maximum accuracy.

Macro F1 averages the F1-score for each class equally. It is useful when
class frequencies differ, but the class-wise confusion matrix should
also be reviewed to understand errors for High, Moderate, and Low
productivity. These are results from one held-out split of the reported
dataset; they are not a guarantee of future performance. Validate the
final dataset and rerun training before using these metrics in a final
report.

The training script writes evaluation outputs to:

-   `backend/artifacts/model_comparison.csv`
-   `backend/artifacts/metadata.json`

It also stores confusion matrices and feature-importance metadata in
`metadata.json`.

## System Architecture

``` text
CSV Dataset
    |
    v
Pandas: Data Understanding and Preprocessing
    |
    v
Feature Engineering + scikit-learn Pipelines
    |
    v
Train/Test Split -> KNN / Random Forest / Decision Tree
    |
    v
Evaluation -> Select Highest Macro F1
    |
    v
Saved Model Artifacts + metadata.json + employees.csv
    |
    v
Django Backend / Prediction Service
    |
    v
Frontend: HTML + CSS + React + Chart.js
    |
    v
Employee Dashboard + Prediction + What-If Simulation
```

### Key modules

-   `backend/train_model.py` --- data analysis, preprocessing, model
    training, evaluation, and artifact generation.
-   `backend/api/features.py` --- target/ID configuration and feature
    engineering.
-   `backend/api/ml_service.py` --- loads metadata and saved models,
    prepares prediction inputs, and returns predictions and
    probabilities.
-   `backend/api/views.py` --- Django view/API handlers.
-   `backend/api/urls.py` --- API routes.
-   `backend/config/settings.py` --- Django settings, static files, and
    database configuration.
-   `backend/config/urls.py` --- routes `/` to the dashboard and `/api/`
    to API routes.
-   `frontend/index.html` --- dashboard HTML entry point.
-   `frontend/app.jsx` --- React dashboard.
-   `frontend/style.css` --- dashboard styling.
## API
| Method | URL | Purpose |
|---|---|---|
| GET | /api/health/ | status + best model |
| GET | /api/schema/ | input features, ranges, classes, models |
| GET | /api/summary/ | EDA stats, metrics, confusion matrices, importance |
| GET | /api/employees/ , /api/employees/<id>/?model= | twin list / profile + prediction + history |
| POST | /api/predict/ `{features, model?, employee_id?}` | prediction + probabilities |
| POST | /api/whatif/ `{base, scenario, model?, employee_id?}` | baseline vs scenario |
| GET | /api/history/?employee_id= | logged predictions |


## Project Structure

``` text
employee-digital-twin/
└── twin/
    ├── README.md
    ├── requirements.txt
    ├── .gitignore
    ├── data/
    │   ├── 06_employee_productivity_digital_twin.csv
    │   └── make_sample_data.py
    ├── backend/
    │   ├── manage.py
    │   ├── train_model.py
    │   ├── config/
    │   │   ├── settings.py
    │   │   ├── urls.py
    │   │   └── wsgi.py
    │   ├── api/
    │   │   ├── features.py
    │   │   ├── ml_service.py
    │   │   ├── views.py
    │   │   ├── urls.py
    │   │   └── migrations/
    │   └── artifacts/
    │       ├── best_model.pkl
    │       ├── knn_model.pkl
    │       ├── random_forest_model.pkl
    │       ├── decision_tree_model.pkl
    │       ├── employees.csv
    │       ├── metadata.json
    │       └── model_comparison.csv
    └── frontend/
        ├── index.html
        ├── app.jsx
        └── style.css
```

## Run Locally

### Prerequisites

-   Python installed.
-   Git, if cloning from GitHub.
-   Internet access for the frontend CDN libraries.

### 1. Clone the repository

``` bash
git clone https://github.com/Rajeshwari0712/employee-digital-twin.git
cd employee-digital-twin/twin
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use the appropriate local
execution-policy guidance for your machine, or run the environment's
Python executable directly.

### 3. Install dependencies

``` powershell
pip install -r requirements.txt
pip install gunicorn whitenoise
```

Gunicorn is primarily needed for production deployment; Windows local
development can use Django's development server.

### 4. Train models and generate artifacts

From the `twin` directory:

``` powershell
python backend/train_model.py data/06_employee_productivity_digital_twin.csv
```

This command assumes the CSV contains the target and input columns
expected by `api/features.py`. If your assigned dataset has different
column names or a different target, update the target/feature
configuration and preprocessing logic first.

### 5. Run Django migrations

``` powershell
cd backend
python manage.py migrate
```

### 6. Start the development server

``` powershell
python manage.py runserver
```

Open `http://127.0.0.1:8000/` in a browser.

## Train the Models on a Dataset

The training script accepts an optional CSV path:

``` bash
python backend/train_model.py path/to/data.csv
```

From inside the `backend` directory, the equivalent command is:

``` bash
python train_model.py path/to/data.csv
```

The script expects the target column and ID column configured in
`api/features.py`. It is not a fully schema-agnostic trainer: a
different dataset may require changes to the target name, ID column,
invalid-value rules, and feature-engineering code. After training, check
the printed metrics and the generated files in `backend/artifacts/`.

## Deployment

The project has been deployed on Render at:

**https://employee-digital-twin.onrender.com**

The repository is hosted at:

**https://github.com/Rajeshwari0712/employee-digital-twin**

The current Django setup serves the dashboard at `/` and includes API
URL patterns under `/api/`. Production configuration should use
environment variables for `SECRET_KEY`, `DEBUG`, and `ALLOWED_HOSTS`.
Keep the secret key private and never commit it to GitHub.

For static files, the project uses Django's static-file settings and
should configure `STATIC_ROOT`, run `collectstatic`, and use WhiteNoise
if serving static files through the Django application. Verify that the
deployed page, CSS, JSX, and API endpoints all load correctly. A Render
deployment marked "Live" means the service started; it does not by
itself guarantee that every page or prediction endpoint works correctly.

The current configuration uses SQLite. For a prototype this may be
sufficient, but local SQLite data should not be assumed to be durable
across redeployments or ephemeral filesystems. If persistent application
records are required, consider a managed database.

## Limitations and Future Scope

### Current limitations

-   The reported dataset has 500 rows; larger and more representative
    data is needed for stronger validation.
-   Results are based on one stratified train-test split.
-   Class imbalance may affect per-class performance; confusion matrices
    and class-level precision/recall should be examined.
-   Model predictions are not causal explanations and should not be used
    alone for consequential employee decisions.
-   The dataset's source and license must be documented by the team.
-   The dashboard should not be described as a real-time synchronized
    twin unless it is connected to a live data source.
-   Persistent historical tracking should only be claimed if predictions
    and state changes are actually stored and retrieved.

### Future enhancements

-   Train and validate on a larger, independently collected dataset.
-   Use cross-validation and additional holdout data to estimate
    performance more robustly.
-   Investigate class imbalance and tune hyperparameters.
-   Add class-wise reports, confusion-matrix visualization, and
    model-calibration checks.
-   Add authentication and role-based access if employee data is
    sensitive.
-   Add a persistent database and prediction-history timeline.
-   Connect to an authorized live data source for more current employee
    state.
-   Build a production frontend bundle instead of relying on
    browser-side Babel and development CDN builds.
-   Add automated tests, input validation, monitoring, and deployment
    checks.

## References

1.  **Project dataset:** `06_employee_productivity_digital_twin.csv`.
    Add the actual dataset publisher/source, URL or citation, license,
    and access date before submission.
2.  **Django documentation:** https://docs.djangoproject.com/
3.  **scikit-learn documentation:** https://scikit-learn.org/stable/
4.  **Pandas documentation:** https://pandas.pydata.org/docs/
5.  **NumPy documentation:** https://numpy.org/doc/
6.  **React documentation:** https://react.dev/
7.  **Chart.js documentation:** https://www.chartjs.org/docs/
8.  **Render documentation:** https://render.com/docs

------------------------------------------------------------------------

**Academic submission note:** Replace or confirm the dataset
description, source/license, college/department details, team member
names, roll numbers, and any implementation details that changed after
this README was prepared. The accompanying project documentation
template requests these items as part of the report.
