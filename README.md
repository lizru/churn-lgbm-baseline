# Baseline Customer Churn Prediction with LightGBM

This project builds a baseline machine learning pipeline to predict customer churn using the Telco Customer Churn dataset from Kaggle. It uses a LightGBM classifier and includes model evaluation with key metrics, visualization, and SHAP explainability.

## Dataset

- The data comes from the [Telco Customer Churn dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

## Features

- Data cleaning and preprocessing pipeline
- LightGBM classifier with basic hyperparameter tuning via randomized search
- Custom threshold selection optimized for F2 score, prioritizing recall
- Model evaluation with accuracy, precision, recall, F1 score, confusion matrix, ROC curve, and precision-recall curve
- SHAP explainability to interpret feature impacts on predictions