"""
Titanic Survival Prediction - Model Training Script
ML Miniproject

Dataset (download and place train.csv in the same folder as this script):
https://www.kaggle.com/competitions/titanic/data
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ----------------------------------------------------------
# STEP 1: Load Data
# ----------------------------------------------------------
df = pd.read_csv("train.csv")
print("Dataset shape:", df.shape)
print(df.head())
print("\nMissing values per column:\n", df.isnull().sum())

# ----------------------------------------------------------
# STEP 2: Feature Engineering
# ----------------------------------------------------------

# Extract Title from Name (Mr, Mrs, Miss, Master, Rare, etc.)
df["Title"] = df["Name"].str.extract(r",\s*([^\.]*)\.")

title_map = {
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
    "Lady": "Rare", "Countess": "Rare", "Capt": "Rare",
    "Col": "Rare", "Don": "Rare", "Dr": "Rare", "Major": "Rare",
    "Rev": "Rare", "Sir": "Rare", "Jonkheer": "Rare", "Dona": "Rare"
}
df["Title"] = df["Title"].replace(title_map)
df["Title"] = df["Title"].apply(
    lambda x: x if x in ["Mr", "Mrs", "Miss", "Master", "Rare"] else "Rare"
)

# Family size and IsAlone flag
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

# HasCabin - the fact that cabin data exists at all is informative
# (mostly recorded for wealthier / higher-class passengers)
df["HasCabin"] = df["Cabin"].notnull().astype(int)

# ----------------------------------------------------------
# STEP 3: Handle Missing Values
# ----------------------------------------------------------

# Age -> fill with median Age per Title group (better than one global median)
df["Age"] = df.groupby("Title")["Age"].transform(lambda x: x.fillna(x.median()))

# Embarked -> fill with mode (most frequent port)
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Fare -> fill with median (safety net, mainly relevant for test.csv)
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# ----------------------------------------------------------
# STEP 4: Drop Irrelevant Columns
# ----------------------------------------------------------
df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)

# ----------------------------------------------------------
# STEP 5: Encode Categorical Variables
# ----------------------------------------------------------
le_sex = LabelEncoder()
df["Sex"] = le_sex.fit_transform(df["Sex"])          # female=0, male=1

le_embarked = LabelEncoder()
df["Embarked"] = le_embarked.fit_transform(df["Embarked"])

le_title = LabelEncoder()
df["Title"] = le_title.fit_transform(df["Title"])

# ----------------------------------------------------------
# STEP 6: Feature Scaling (standardization)
# ----------------------------------------------------------
scaler = StandardScaler()
numeric_cols = ["Age", "Fare", "FamilySize"]
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# ----------------------------------------------------------
# STEP 7: Train-Test Split
# ----------------------------------------------------------
X = df.drop(columns=["Survived"])
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------------------------------------
# STEP 8: Train Multiple Models & Compare
# ----------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results[name] = {"model": model, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    print(f"\n----- {name} -----")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# ----------------------------------------------------------
# STEP 9: Select Best Model (highest accuracy) & Save
# ----------------------------------------------------------
best_name = max(results, key=lambda k: results[k]["accuracy"])
best_model = results[best_name]["model"]
best_acc = results[best_name]["accuracy"]

print(f"\nBest Model: {best_name} (Accuracy: {best_acc:.4f})")

joblib.dump(best_model, "titanic_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le_sex, "le_sex.pkl")
joblib.dump(le_embarked, "le_embarked.pkl")
joblib.dump(le_title, "le_title.pkl")
joblib.dump(best_acc, "model_accuracy.pkl")

print("\nModel and preprocessing objects saved successfully:")
print("  titanic_model.pkl, scaler.pkl, le_sex.pkl, le_embarked.pkl, le_title.pkl, model_accuracy.pkl")