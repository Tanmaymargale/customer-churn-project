# 💼 Customer Churn Prediction System

## Overview
This is an end-to-end Machine Learning web application for **Customer Churn Prediction**.  
Users can train a model, test it on new data, and predict churn for individual customers.

**Skills Covered During Internship:**
- Python Programming
- Machine Learning (Supervised Learning)
- Model Training, Evaluation, and Prediction
- XGBoost Model
- FastAPI Backend Development
- REST API Design
- Streamlit Frontend Development
- GitHub Project Management & Documentation

---

## Project Structure

---
ustomer-churn-project/
├── backend/ # FastAPI backend code
├── frontend/ # Streamlit frontend
├── dataset/ # Sample training and testing CSV files
├── models/ # Saved ML models
├── requirements.txt
└── README.md


---

## Features

### 1️⃣ Train Model
- Upload CSV dataset
- Train a RandomForest model with preprocessing
- Returns number of rows used

### 2️⃣ Test Model
- Upload test CSV
- Returns metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score

### 3️⃣ Predict Single Customer
- Input individual customer data
- Returns churn prediction and meaning
- Download prediction CSV

---

## Backend API Endpoints

| Endpoint      | Method | Description |
|---------------|--------|-------------|
| `/train`      | POST   | Train model using uploaded CSV |
| `/test`       | POST   | Test model using uploaded CSV |
| `/predict`    | POST   | Predict single customer churn |

---

## Frontend (Streamlit)

- Interactive sidebar for navigation
- Progress bars and metrics cards
- Colorful, real-world interface
- User-friendly forms for prediction
- Download prediction as CSV

---

## Installation

1. Clone the repo:
```bash
git clone <your-repo-link>
cd customer-churn-project

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

uvicorn backend.main:app --reload

streamlit run frontend/app.py