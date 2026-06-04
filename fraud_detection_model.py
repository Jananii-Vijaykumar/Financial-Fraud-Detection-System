import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import os

# Load Dataset
df = pd.read_csv("credit_card_fraud_dataset.csv")

# Convert text columns to numbers
le_type = LabelEncoder()
le_location = LabelEncoder()

df["TransactionType"] = le_type.fit_transform(df["TransactionType"])
df["Location"] = le_location.fit_transform(df["Location"])

# Remove date column
df = df.drop("TransactionDate", axis=1)

# Features and Target
X = df.drop("IsFraud", axis=1)
y = df["IsFraud"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print("Accuracy:", accuracy)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, pred)

print("Confusion Matrix:")
print(cm)

# Save confusion matrix
joblib.dump(cm, "models/confusion_matrix.pkl")

# Create folder if missing
os.makedirs("models", exist_ok=True)

# Save model
joblib.dump(model, "models/fraud_model.pkl")

print("Model saved successfully!")