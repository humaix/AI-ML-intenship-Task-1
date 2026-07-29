# Kidney Disease Classification – End-to-End Data Science Pipeline

## Research Internship Program 2026 – Batch 2
**Alphatron Technologies × MMSELab (NED University)**

---

# Project Overview

This project implements a complete end-to-end Data Science Pipeline for Chronic Kidney Disease (CKD) prediction using Machine Learning.

The objective of this project is to demonstrate a production-oriented AI/ML workflow by following software engineering best practices such as modular design, Object-Oriented Programming (OOP), reusable components, configuration management, model persistence, version control, and project documentation.

The entire pipeline has been implemented using Python and Scikit-learn while maintaining a clean and maintainable project architecture.

---

# Task Objectives

This project was developed to satisfy the internship requirements by implementing the following stages:

- ✔ Data Collection
- ✔ Data Loading
- ✔ Data Understanding
- ✔ Data Cleaning
- ✔ Data Preprocessing
- ✔ Exploratory Data Analysis (EDA)
- ✔ Feature Engineering
- ✔ Data Splitting
- ✔ Model Selection
- ✔ Model Training
- ✔ Hyperparameter Tuning
- ✔ Model Evaluation
- ✔ Model Comparison
- ✔ Model Saving
- ✔ Prediction / Inference
- ✔ Result Visualization
- ✔ Documentation

---

# Dataset Information

### Dataset

Kidney Disease Dataset

### Problem Type

Binary Classification

### Target Variable

| Value | Meaning |
|-------|----------|
| 1 | Chronic Kidney Disease (CKD) |
| 0 | Healthy (Not CKD) |

The dataset contains patient medical information including laboratory test results and clinical observations used to predict kidney disease.

---
## Project Architecture

```text
Task1 ML pipeline
│
├── .github
│   └── workflows
│       └── python-ci.yml
│
├── config
│   └── config.yaml
│
├── data
│   ├── raw
│   └── processed
│
├── models
│   └── kidney_disease_model.joblib
│
├── notebooks
│   └── eda.ipynb
│
├── output
│   └── figures
│
├── reports
│   ├── evaluation.json
│   └── figures
│
├── src
│   ├── data_loader.py
│   ├── data_cleaner.py
│   ├── feature_engineer.py
│   ├── preprocessor.py
│   ├── model_trainer.py
│   ├── hyperparameter_tuner.py
│   ├── model_evaluator.py
│   └── pipeline_runner.py
│
├── predict.py
├── run_pipeline.py
├── requirements.txt
├── README.md
└── .gitignore
```


# Data Science Pipeline

The implemented workflow follows the complete machine learning lifecycle.

```
Raw Dataset
      │
      ▼
Data Loading
      │
      ▼
Data Understanding
      │
      ▼
Data Cleaning
      │
      ▼
Feature Engineering
      │
      ▼
Preprocessing
      │
      ▼
Train/Test Split
      │
      ▼
Model Training
      │
      ▼
Hyperparameter Tuning
      │
      ▼
Model Evaluation
      │
      ▼
Model Comparison
      │
      ▼
Best Model Selection
      │
      ▼
Model Saving
      │
      ▼
Prediction / Inference
```

---

# Object-Oriented Design

The project follows a modular Object-Oriented Programming (OOP) architecture.

| Module | Responsibility |
|---------|---------------|
| DataLoader | Load dataset |
| DataCleaner | Clean missing values and invalid data |
| FeatureEngineer | Create additional features |
| DataPreprocessor | Encode and scale features |
| ModelTrainer | Train machine learning models |
| HyperparameterTuner | Tune Random Forest parameters |
| ModelEvaluator | Evaluate and compare models |
| PipelineRunner | Execute complete pipeline |

---

# Exploratory Data Analysis (EDA)

The notebook performs:

- Dataset overview
- Missing value analysis
- Target class distribution
- Numerical feature distributions
- Correlation Heatmap
- Boxplots grouped by target class

Generated visualizations are automatically stored inside:

```
output/figures/
```

---

# Feature Engineering

Additional features created include:

- Age Group
- Blood Glucose to Hemoglobin Ratio
- Kidney Function Score

These engineered features improve the predictive capability of the models.

---

# Machine Learning Models

The following models were trained and compared:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Tuned Random Forest

---

# Hyperparameter Tuning

GridSearchCV was used to optimize the Random Forest classifier by searching multiple parameter combinations to improve predictive performance.

---

# Evaluation Metrics

Each model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix
- ROC Curve

Evaluation metrics are saved in:

```
reports/evaluation.json
```

Generated evaluation figures are stored in:

```
reports/figures/
```

---

# Model Persistence

The best-performing model is saved using Joblib.

```
models/
└── kidney_disease_model.joblib
```

This model is later used for prediction without retraining.

---

# Prediction

Predictions can be made using:

```bash
python predict.py --input "<patient_json>"
```

Example output:

```
Diagnosis: CKD

Probability of CKD: 0.96
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/humaix/AI-ML-intenship-Task-1.git
```

Move into the project

```bash
cd AI-ML-intenship-Task-1
```

---

# Create Virtual Environment

Windows

```bash
python -m venv .venv
```

Activate

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

Execute the complete pipeline

```bash
python run_pipeline.py
```

Run prediction

```bash
python predict.py --input "<patient_json>"
```

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- YAML
- Git
- GitHub
- Jupyter Notebook

---

# Software Engineering Practices

This project follows professional software engineering principles required by the internship.

- Object-Oriented Programming (OOP)
- Modular Architecture
- Clean Code Principles
- Configuration File Management
- Version Control using Git
- GitHub Repository Management
- Model Serialization
- Documentation
- Reproducible Pipeline

---

# Results

The project successfully implements a complete production-style machine learning workflow from raw data to prediction.

The trained models were evaluated using multiple performance metrics, and the best-performing model was saved for future inference. Visualization reports and evaluation metrics are automatically generated, making the pipeline suitable for experimentation and further development.

---
## CI/CD Pipeline

This project includes a basic GitHub Actions CI pipeline.

The workflow automatically:

- Checks out the repository
- Sets up Python
- Installs project dependencies
- Verifies that all Python source files compile successfully

The workflow runs automatically whenever code is pushed to the repository or a pull request is created.
# Future Improvements

- CI/CD Pipeline using GitHub Actions
- Automated Unit Testing
- Streamlit Web Application
- Docker Containerization
- Explainable AI (SHAP/LIME)
- Cloud Deployment
- Model Monitoring

---

# Author

**Humaiz Ahmed**

BS Computer Science (Artificial Intelligence)

Research Internship Program 2026 – Batch 2

Alphatron Technologies × MMSELab (NED University)