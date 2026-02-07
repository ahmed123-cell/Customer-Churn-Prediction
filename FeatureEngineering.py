from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.internet_weight = {
            'Fiber optic': 3,
            'DSL': 2,
            'No': 1
        }
        self.contract_risk = {
            'Month-to-month': 3,
            'One year': 2,
            'Two year': 1
        }

    def fit(self, X, y=None):
        return self   # Nothing to learn

    def transform(self, X):
        X = X.copy()

        # Create Feature 1: Internet burden
        X['Internet_burden'] = (X['InternetService'].map(self.internet_weight) * X['MonthlyCharges'] / (X['tenure'] + 1))

        # Create Feature 2: price per month
        X['Price_per_month'] = X['TotalCharges'] / (X['tenure'] + 1)

        # Create Feature 3: risk score
        X['Contract_risk'] = X['Contract'].map(self.contract_risk)
        X['Risk_score'] = X['Contract_risk'] * X['MonthlyCharges']

        return X