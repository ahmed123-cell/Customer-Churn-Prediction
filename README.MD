# Telco Customer Churn Prediction API 📞

A FastAPI service that predicts whether a telecom customer will churn using XGBoost.

### Final Performance (Test Set)

- **Recall** (churn = Yes): **0.82**  
- Precision: 0.51  
- F1-score: 0.63  
- ROC AUC: 0.847  
- Accuracy: 0.74  

### Project Structure

```
telco-churn-api/
├── app.py                                   # FastAPI app
├── feature_engineering.py                    # Custom feature transformer
├── requirements.txt
├── xgboost.pkl                              # Saved model + pipeline
├── customer_churn_analysis.ipynb                     # Development notebook
└── README.md
├── customer_churn_features.py                        #  features of the data
```

### Quick Start (Local)

```bash
# Clone
git clone https://github.com/ahmed123-cell/Customer-Churn-Prediction.git
cd telco-churn-api

# Virtual environment
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# Install
pip install -r requirements.txt

# Run
uvicorn main:app --reload
```

→ Open: http://127.0.0.1:8000/docs

**Endpoint**:  
POST `/predict`

**Example request body**:

```json
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
```

> Note: First request after inactivity may take 10–40 seconds (free tier sleeps).
