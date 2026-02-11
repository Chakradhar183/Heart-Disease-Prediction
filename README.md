# 🫀 Heart Disease Prediction System

A production-ready Machine Learning web application that predicts the risk of heart disease using clinical and lifestyle indicators.
The system trains multiple ML models, selects the best-performing one using hyperparameter tuning, and deploys it through a Flask-based web interface and REST API.

Project link: https://heartwebapp.onrender.com

## 📌 Project Overview

Cardiovascular diseases are one of the leading causes of death globally. This project leverages machine learning to analyze health-related features and predict whether a person is at risk of heart disease.

The system:

- Trains and compares multiple ML algorithms
- Performs preprocessing and feature encoding
- Selects the best-performing model automatically
- Deploys the trained model via a Flask web app
- Exposes a REST API endpoint for integration

## 🧪 Technologies Used

- **Python 3.11**
- **Flask**
- **scikit-learn**
- **XGBoost**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Gunicorn** (Production server)
- **Render** (Cloud deployment)

## ✨ Features

### 🔹 Machine Learning

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Random Forest
- Naive Bayes
- XGBoost
- Hyperparameter tuning
- Automatic best model selection

### 🔹 Web Application

- User-friendly form interface
- Real-time prediction
- Confidence score display
- Clean HTML templates

### 🔹 REST API

- `/predict` endpoint for programmatic access
- Accepts JSON input
- Returns JSON prediction output

## 📊 Dataset Information

**Dataset:** Heart Disease 2020 Cleaned Dataset

- 300,000+ records
- 18 health features
- Lifestyle indicators
- Demographic data
- Medical conditions

**Example Features:**

- Age Category
- BMI
- Smoking
- Alcohol Drinking
- Physical Activity
- General Health
- Diabetes
- Stroke
- Sex

## 🧠 Machine Learning Pipeline

### 1️⃣ Data Preprocessing

- Handling categorical features using Label Encoding
- Feature scaling using StandardScaler
- Train-test split

### 2️⃣ Model Training

- Multiple models trained
- Cross-validation
- Accuracy comparison
- Best model selection

### 3️⃣ Model Saving

After training:

- `heartdisease_model.pkl`
- `scaler.pkl`
- `encoders.pkl`

These are used by the Flask application for prediction.

## 🏗️ Project Structure

```
Heart-Disease-Prediction/
│
├── app.py                      # Flask application
├── heartdisease.py             # Model training script
├── heart_2020_cleaned.csv      # Dataset
├── heartdisease_model.pkl      # Saved best model
├── scaler.pkl                  # Scaler object
├── encoders.pkl                # Encoders
│
├── templates/
│   ├── index.html              # Input form
│   └── result.html             # Output page
│
├── static/                     # CSS files
│
├── requirements.txt
├── render.yaml
├── runtime.txt
└── README.md
```
