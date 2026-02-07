from fastapi import FastAPI
import uvicorn
import joblib
import pandas as pd
from Customer_churn import CustomerChurnFeatures
from FeatureEngineering import FeatureEngineering

# load model
model = joblib.load("xgboost.pkl")

# Build API
app = FastAPI(title="Student Depression Prediction")

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running"}

@app.post("/predict")
def prediction(features: CustomerChurnFeatures):

    # Build the DataFrame
    data = pd.DataFrame([{
        "gender": features.gender,
        "SeniorCitizen": features.SeniorCitizen,
        "Partner": features.Partner,
        "Dependents": features.Dependents,
        "tenure": features.tenure,
        "PhoneService": features.PhoneService,
        "MultipleLines": features.MultipleLines,
        "InternetService": features.InternetService,
        "OnlineSecurity": features.OnlineSecurity,
        "OnlineBackup": features.OnlineBackup,
        "DeviceProtection": features.DeviceProtection,
        "TechSupport": features.TechSupport,
        "StreamingTV": features.StreamingTV,
        "StreamingMovies": features.StreamingMovies,
        "Contract": features.Contract,
        "PaperlessBilling": features.PaperlessBilling,
        "PaymentMethod": features.PaymentMethod,
        "MonthlyCharges": features.MonthlyCharges
    }])

    # Create total charges features
    data['TotalCharges'] = data['MonthlyCharges'] * data['tenure']

    # add the new features
    data_transformed = FeatureEngineering().transform(data)
    
    pred = model.predict(data_transformed)
    return {"predicted_result": float(pred[0])}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)


{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "Yes",
  "tenure": 0,
  "PhoneService": "No",
  "MultipleLines": "No",
  "InternetService": "DSL",
  "OnlineSecurity": "Yes",
  "OnlineBackup": "No",
  "DeviceProtection": "Yes",
  "TechSupport": "Yes",
  "StreamingTV": "Yes",
  "StreamingMovies": "No",
  "Contract": "Two year",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Bank transfer (automatic)",
  "MonthlyCharges": 52.55
}