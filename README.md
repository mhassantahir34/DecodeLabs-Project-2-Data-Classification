# DecodeLabs Project 2 - Data Classification Using AI

A supervised machine learning classification project developed using Python as part of my DecodeLabs AI Internship.

## Project Overview

The purpose of this project is to classify e-commerce orders based on different order-related features.

The project uses a dataset containing 1,200 records and applies machine learning classification algorithms to predict the `OrderStatus` of an order.

## Objectives

- Load and understand the dataset
- Handle missing values
- Perform data preprocessing
- Split the dataset into training and testing sets
- Train classification models
- Compare model performance
- Evaluate classification results
- Test the model on new data

## Dataset

The dataset contains information about customer orders, including:

- Product
- Quantity
- Unit Price
- Payment Method
- Order Status
- Items in Cart
- Coupon Code
- Referral Source
- Total Price

### Target Variable

`OrderStatus`

The target contains five classes:

- Cancelled
- Returned
- Pending
- Shipped
- Delivered

## Machine Learning Models

The project compares multiple classification algorithms:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. K-Nearest Neighbors

## Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Excel Dataset
- VS Code

## Project Structure

```text
DecodeLabs-Project-2-Data-Classification/
│
├── P2 Data Classification.py
├── dataset.xlsx
├── model_comparison.csv
├── confusion_matrix.png
├── feature_importance.png
└── README.md
