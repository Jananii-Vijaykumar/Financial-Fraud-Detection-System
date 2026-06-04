# Financial-Fraud-Detection-System
An end-to-end Financial Fraud Detection project featuring data preprocessing, fraud prediction, performance evaluation, and an interactive Streamlit dashboard for transaction analysis.
# Financial Fraud Detection System

## Overview

The Financial Fraud Detection System is a Machine Learning-based application designed to identify fraudulent financial transactions. The system analyzes transaction details and predicts whether a transaction is legitimate or fraudulent using a trained Random Forest Classifier.

## Features

* Fraud Detection using Machine Learning
* Interactive Streamlit Dashboard
* Dataset Preview
* Total Records Display
* Fraud Distribution Visualization
* Confusion Matrix Analysis
* Transaction Fraud Prediction
* Model Performance Evaluation

## Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* Joblib

## Dataset Information

The dataset contains transaction details such as:

* Transaction ID
* Transaction Date
* Transaction Amount
* Merchant ID
* Transaction Type
* Location
* Fraud Status (IsFraud)

## Machine Learning Workflow

1. Data Collection
2. Data Preprocessing
3. Label Encoding
4. Train-Test Split
5. Model Training using Random Forest Classifier
6. Model Evaluation
7. Confusion Matrix Generation
8. Dashboard Deployment using Streamlit

## Model Performance

* Algorithm: Random Forest Classifier
* Accuracy: 98.93%

## Dashboard Components

### Dataset Preview

Displays the first 10 records of the dataset along with the total number of records.

### Fraud Distribution

Visualizes the number of fraudulent and legitimate transactions.

### Confusion Matrix

Displays the model's classification performance using a heatmap.

### Fraud Prediction

Allows users to enter transaction details and receive a fraud prediction.

## Project Structure

Financial_Fraud_Detection/

├── app.py

├── fraud_detection_model.py

├── requirements.txt

├── credit_card_fraud_dataset.csv

├── models/

│ ├── fraud_model.pkl

│ └── confusion_matrix.pkl

└── README.md

## Installation

### Clone Repository

```bash
git clone <repository-url>
cd Financial_Fraud_Detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train the Model

```bash
python fraud_detection_model.py
```

### Run the Application

```bash
python -m streamlit run app.py
```

## Future Enhancements

* Compare Multiple Machine Learning Algorithms
* Real-Time Fraud Detection
* Advanced Feature Engineering
* Model Optimization
* Cloud Deployment
* User Authentication System

## Author
Sharini Pooja
https://github.com/sharinipooja

Jananii
https://github.com/Jananii-Vijaykumar

## Conclusion

This project demonstrates the application of Machine Learning and Data Analytics techniques for detecting fraudulent financial transactions. The system provides an easy-to-use dashboard for transaction analysis and fraud prediction, helping organizations identify suspicious activities efficiently.
