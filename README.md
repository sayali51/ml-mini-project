# 🚢 Titanic Survival Prediction using Machine Learning

A complete end-to-end ML mini project: data cleaning → feature engineering → model comparison → interactive **Streamlit** web app that predicts whether a passenger would have survived the Titanic disaster.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)

---

## 🎬 Demo

> **Live app:** [ADD YOUR STREAMLIT CLOUD LINK HERE, or delete this line]

![App Demo](screenshots/demo.gif)

| Prediction: Survived | Prediction: Did Not Survive |
|---|---|
| ![Survived](screenshots/app_survived.png) | ![Not Survived](screenshots/app_not_survived.png) |

---

## 📌 Problem Statement

Given a passenger's details (class, gender, age, family size, fare, port of embarkation, title), predict whether they **survived (1)** or **did not survive (0)**. This is a binary classification problem using the Kaggle *Titanic: Machine Learning from Disaster* dataset (891 passengers).

---

## ✨ Features

- Data cleaning with smart missing-value handling (Age imputed using median per passenger title)
- Feature engineering: `Title`, `FamilySize`, `IsAlone`, `HasCabin`
- Comparison of 3 models: Logistic Regression, Decision Tree, Random Forest
- Evaluation using Accuracy, Precision, Recall, F1-score and Confusion Matrix
- 7 EDA charts auto-generated
- Interactive Streamlit app with prediction, confidence score and probability chart

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Logistic Regression | [XX]% | [XX] | [XX] | [XX] |
| Decision Tree | [XX]% | [XX] | [XX] | [XX] |
| **Random Forest** | **[XX]%** | [XX] | [XX] | [XX] |

> Fill these from the output of `python train_model.py`. Mark the best model in bold.

![Model Output](screenshots/model_output.png)

---

## 🔍 Exploratory Data Analysis

| Survival by Gender | Survival by Class |
|---|---|
| ![Gender](eda_charts/2_survival_by_gender.png) | ![Class](eda_charts/3_survival_by_class.png) |

| Age Distribution | Correlation Heatmap |
|---|---|
| ![Age](eda_charts/4_age_distribution.png) | ![Heatmap](eda_charts/6_correlation_heatmap.png) |

**Key insights:** women and first-class passengers had much higher survival rates; children also had better chances, matching the historical "women and children first" evacuation rule.

---

## 🗂️ Project Structure

```
titanic-survival-prediction/
├── train.csv                 # Kaggle dataset (download separately, see below)
├── train_model.py            # preprocessing + training + evaluation + saves model
├── eda.py                    # generates EDA charts into eda_charts/
├── app.py                    # Streamlit web app
├── view_saved_files.py       # prints contents of saved model/encoders
├── requirements.txt
├── titanic_model.pkl         # trained model (generated)
├── scaler.pkl                # fitted scaler (generated)
├── le_sex.pkl                # label encoders (generated)
├── le_embarked.pkl
├── le_title.pkl
├── model_accuracy.pkl
├── eda_charts/               # EDA charts (generated)
├── screenshots/              # demo images used in this README
└── README.md
```

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/sayali51/ml-mini-project.git
cd ml-mini-project
```

**2. Install dependencies**
```bash
python -m pip install -r requirements.txt
```

**3. Download the dataset**
Download `train.csv` from [Kaggle Titanic](https://www.kaggle.com/competitions/titanic/data) and place it in the project folder.

**4. Train the model**
```bash
python train_model.py
```

**5. (Optional) Generate EDA charts**
```bash
python eda.py
```

**6. Launch the web app**
```bash
python -m streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 🧪 Try These Sample Passengers

| Field | Example 1 | Example 2 |
|---|---|---|
| Passenger Class | 1 | 3 |
| Gender | female | male |
| Age | 28 | 30 |
| Siblings/Spouses | 0 | 0 |
| Parents/Children | 0 | 0 |
| Fare | 100 | 8 |
| Embarked | C | S |
| Title | Miss | Mr |
| Recorded Cabin | Yes | No |
| **Expected result** | Survived | Did Not Survive |

---

## 🛠️ Tech Stack

Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · Streamlit · Joblib

---

## 🔮 Future Scope

- Hyperparameter tuning with GridSearchCV
- More features (fare per person, deck from Cabin)
- Gradient boosting models (XGBoost, LightGBM)
- SHAP explanations for individual predictions

---

## 👩‍💻 Author

**Sayali** — Computer Engineering, AISSMS College of Engineering, Pune (SPPU)
GitHub: [@sayali51](https://github.com/sayali51)

---

## 📄 License

For educational purposes. Dataset © Kaggle.
