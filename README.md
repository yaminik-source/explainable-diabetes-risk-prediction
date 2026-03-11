# Explainable Diabetes Risk Prediction using Machine Learning

## Overview

This project develops an interpretable machine learning model to predict diabetes risk based on clinical health indicators. The objective is to explore how explainable AI techniques can support healthcare prediction tasks.

The model is trained on the Pima Indians Diabetes Dataset and evaluated using ROC-AUC metrics.

## Key Features

• Exploratory Data Analysis  
• Feature correlation analysis  
• Random Forest classification model  
• Feature importance analysis  
• SHAP-based model interpretability  

## Project Structure

data/  
Contains raw and processed datasets.

notebooks/  
Exploratory data analysis and visualization.

src/  
Python scripts for data preprocessing, model training, and evaluation.

results/  
Generated plots and evaluation metrics.

models/  
Trained machine learning models.

## Technologies Used

Python  
Pandas  
NumPy  
Scikit-learn  
Matplotlib  
Seaborn  
SHAP  

## Objective

To build a transparent machine learning pipeline that balances predictive performance with interpretability in healthcare applications.
## Model Performance

The Random Forest model was evaluated using ROC-AUC.

| Metric | Score |
|------|------|
| ROC-AUC | 0.83 |
| Train-Test Split | 80 / 20 |
| Algorithm | Random Forest |

The model demonstrates strong predictive performance for identifying diabetes risk using clinical health indicators.

Key influential features identified by the model include:

- Glucose
- BMI
- Age
- Diabetes Pedigree Function

## Future Work

Future work will explore deep learning models and causal inference techniques to improve prediction reliability in healthcare datasets.
