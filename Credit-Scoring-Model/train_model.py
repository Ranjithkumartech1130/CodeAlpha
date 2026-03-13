import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Sample dataset
data = {
    "income": [50000, 60000, 25000, 40000, 80000, 30000],
    "debt": [10000, 20000, 15000, 5000, 25000, 20000],
    "payment_history": [1, 1, 0, 1, 1, 0],
    "credit_utilization": [0.3, 0.4, 0.8, 0.2, 0.5, 0.9],
    "credit_years": [5, 7, 2, 4, 10, 1],
    "credit_status": [1, 1, 0, 1, 1, 0]
}

df = pd.DataFrame(data)

X = df.drop("credit_status", axis=1)
y = df["credit_status"]

model = RandomForestClassifier()
model.fit(X, y)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model Saved Successfully!")
