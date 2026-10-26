"""
Titanic Survival Prediction - Exploratory Data Analysis (EDA)
Run this AFTER placing train.csv in the same folder.
Saves all charts as PNG files in an 'eda_charts' folder.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("eda_charts", exist_ok=True)
sns.set_style("whitegrid")

df = pd.read_csv("train.csv")

print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nSummary statistics:\n", df.describe())
print("\nSurvival rate overall:", df["Survived"].mean())

# 1. Overall survival count
plt.figure(figsize=(5, 4))
sns.countplot(data=df, x="Survived")
plt.title("Overall Survival Count (0 = Did Not Survive, 1 = Survived)")
plt.savefig("eda_charts/1_overall_survival.png", bbox_inches="tight")
plt.close()

# 2. Survival by Gender
plt.figure(figsize=(5, 4))
sns.barplot(data=df, x="Sex", y="Survived")
plt.title("Survival Rate by Gender")
plt.savefig("eda_charts/2_survival_by_gender.png", bbox_inches="tight")
plt.close()

# 3. Survival by Passenger Class
plt.figure(figsize=(5, 4))
sns.barplot(data=df, x="Pclass", y="Survived")
plt.title("Survival Rate by Passenger Class")
plt.savefig("eda_charts/3_survival_by_class.png", bbox_inches="tight")
plt.close()

# 4. Age distribution by Survival
plt.figure(figsize=(6, 4))
sns.histplot(data=df, x="Age", hue="Survived", kde=True, bins=30, multiple="stack")
plt.title("Age Distribution by Survival")
plt.savefig("eda_charts/4_age_distribution.png", bbox_inches="tight")
plt.close()

# 5. Survival by Embarkation Port
plt.figure(figsize=(5, 4))
sns.barplot(data=df, x="Embarked", y="Survived")
plt.title("Survival Rate by Port of Embarkation")
plt.savefig("eda_charts/5_survival_by_embarked.png", bbox_inches="tight")
plt.close()

# 6. Correlation heatmap (numeric columns only)
plt.figure(figsize=(7, 6))
numeric_df = df.select_dtypes(include=["int64", "float64"])
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.savefig("eda_charts/6_correlation_heatmap.png", bbox_inches="tight")
plt.close()

# 7. Fare distribution by Survival
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="Survived", y="Fare")
plt.title("Fare Distribution by Survival")
plt.savefig("eda_charts/7_fare_boxplot.png", bbox_inches="tight")
plt.close()

print("\nAll EDA charts saved in the 'eda_charts' folder.")