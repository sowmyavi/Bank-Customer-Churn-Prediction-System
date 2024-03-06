# Bank-Customer-Churn-Prediction-System

## Project Overview
This project aims to predict bank customer churn by analyzing customer demographics, account information, and transaction behaviors. Using a dataset containing 10,000 customers' details across 14 variables, we develop models to identify key factors contributing to customer turnover and predict future churn.

## Dataset Description
The dataset includes 10,000 records with 14 columns, capturing customer demographics (age, gender, nationality), financial behaviors (balance, credit score, product usage), and account status (active membership, estimated salary). The goal is to use this information to predict whether a customer will leave the bank.

## Exploratory Data Analysis (EDA)
- **Distribution Analysis:** Examines the spread of demographic and financial attributes across the customer base.
- **Feature Correlation:** Investigates how different customer attributes relate to churn.
- **Customer Behavior Insights:** Identifies patterns and trends that influence churn decisions.

## Key Findings
- Age, number of products used, and account activity status significantly impact customer churn.
- Customers with zero balance and those with balances between 100,000 to 150,000 show higher churn rates.
- Gender and estimated salary exhibit minimal influence on churn predictions.

## Models Used
- Decision Tree Classifier
- Random Forest Classifier

Both models underwent hyperparameter tuning via GridSearchCV, with Random Forest Classifier achieving superior accuracy and precision.
