from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
from io import StringIO
from pathlib import Path

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import numpy as np

app = FastAPI(title="Customer Churn Prediction API - Enhanced")

# -------------------------------
# Paths & columns
# -------------------------------
MODEL_DIR = Path("../models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "churn_pipeline_xgb.pkl"

TARGET = "churn"
DROP_COLUMNS = ["customer_id"]

# -------------------------------
# Pydantic Model for Predict Route
# -------------------------------
class CustomerData(BaseModel):
    credit_score: int = Field(..., example=650, description="Credit score of the customer")
    country: str = Field(..., example="France", description="Customer's country")
    gender: str = Field(..., example="Male", description="Customer's gender")
    age: int = Field(..., example=35, description="Customer's age")
    tenure: int = Field(..., example=5, description="Number of years as a customer")
    balance: float = Field(..., example=50000, description="Account balance")
    products_number: int = Field(..., example=2, description="Number of products used")
    credit_card: int = Field(..., example=1, description="Has credit card? 1 = Yes, 0 = No")
    active_member: int = Field(..., example=1, description="Active member? 1 = Yes, 0 = No")
    estimated_salary: float = Field(..., example=60000, description="Estimated salary")

# -------------------------------
# Pipeline builder
# -------------------------------
def build_pipeline():
    categorical_cols = ["country", "gender"]
    numeric_cols = [
        "credit_score", "age", "tenure",
        "balance", "products_number",
        "credit_card", "active_member",
        "estimated_salary"
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

    xgb_model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
        scale_pos_weight=1.5  # Adjust for class imbalance
    )

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", xgb_model)
    ])

    return pipeline

# -------------------------------
# TRAIN ENDPOINT
# -------------------------------
@app.post("/train")
async def train_model(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode()))
        df.columns = df.columns.str.strip()
        df = df.drop(columns=DROP_COLUMNS, errors="ignore")

        if TARGET not in df.columns:
            raise HTTPException(400, f"Target column '{TARGET}' not found in CSV.")

        X = df.drop(columns=[TARGET])
        y = df[TARGET]

        pipeline = build_pipeline()
        pipeline.fit(X, y)

        joblib.dump(pipeline, MODEL_PATH)

        return {"status": "Training completed", "rows_used": len(df)}

    except Exception as e:
        raise HTTPException(500, f"Error during training: {str(e)}")

# -------------------------------
# TEST ENDPOINT
# -------------------------------
@app.post("/test")
async def test_model(file: UploadFile = File(...)):
    try:
        if not MODEL_PATH.exists():
            raise HTTPException(400, "Model not found. Train the model first.")

        pipeline = joblib.load(MODEL_PATH)

        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode()))
        df.columns = df.columns.str.strip()
        df = df.drop(columns=DROP_COLUMNS, errors="ignore")

        if TARGET not in df.columns:
            raise HTTPException(400, f"Target column '{TARGET}' not found in CSV.")

        X_test = df.drop(columns=[TARGET])
        y_test = df[TARGET]

        preds = pipeline.predict(X_test)

        metrics = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds)),
            "recall": float(recall_score(y_test, preds)),
            "f1_score": float(f1_score(y_test, preds))
        }

        return {"status": "Testing completed", "metrics": metrics}

    except Exception as e:
        raise HTTPException(500, f"Error during testing: {str(e)}")

# -------------------------------
# PREDICT ENDPOINT (with probabilities)
# -------------------------------
@app.post("/predict")
async def predict(customer: CustomerData):
    try:
        if not MODEL_PATH.exists():
            raise HTTPException(400, "Model not found. Train the model first.")

        pipeline = joblib.load(MODEL_PATH)
        df = pd.DataFrame([customer.dict()])
        df = df.drop(columns=DROP_COLUMNS, errors="ignore")

        prediction = pipeline.predict(df)[0]
        probability = pipeline.predict_proba(df)[0][1]  # probability of churn

        return {
            "prediction": int(prediction),
            "probability": round(float(probability)*100, 2),  # as percentage
            "meaning": "Customer will churn" if prediction == 1 else "Customer will stay"
        }

    except Exception as e:
        raise HTTPException(500, f"Error during prediction: {str(e)}")

# -------------------------------
# ROOT CHECK
# -------------------------------
@app.get("/")
def home():
    return {"message": "Customer Churn Backend Running"}