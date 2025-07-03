# Baseline Customer Churn Prediction with LightGBM

This project builds a baseline machine learning pipeline to predict customer churn. It uses a LightGBM classifier and includes model evaluation with key metrics, visualization, and SHAP explainability.

## Use

Because the model was optimized to prioritize recall through hyperparameter and threshold tuning, it catches most churners but at the cost of many false alarms. In a business setting, this model could be used for customer retention programs; however, the high presence of false alarms means that there is a chance of wasting resources if marketing to all customers identified as being at risk of churning. Therefore, this model would be best suited towards low-cost retention efforts, such as small discount offers or creating targeted email marketing channels.

## Dataset

- The data comes from the [Telco Customer Churn dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

## Features

- Data cleaning and preprocessing pipeline
- LightGBM classifier with basic hyperparameter tuning via randomized search
- Custom threshold selection optimized for F2 score, prioritizing recall
- Model evaluation with accuracy, precision, recall, F1 score, confusion matrix, ROC curve, and precision-recall curve
- SHAP explainability to interpret feature impacts on predictions

