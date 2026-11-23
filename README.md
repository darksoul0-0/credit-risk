# 💳 Credit Risk Prediction & Analysis

<p align="center">
  <b>Turning loan applicant data into smarter, lower-risk lending decisions.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-Data%20Analysis-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/Streamlit-Interactive%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter">
</p>

---

## 📌 About the Project

**Credit Risk Prediction & Analysis** is an end-to-end machine learning project that analyzes **[N] loan applications** to understand what drives loan default and to predict which applicants are most likely to default.

The project transforms raw loan data into a working risk scoring tool using:

**Python → Data Cleaning & Feature Engineering → EDA → Machine Learning → Streamlit App**

The goal is to help lenders make better decisions around **loan approval, risk-based pricing, and portfolio risk management**.

---

## 🎯 Business Objectives

This project focuses on answering questions such as:

* ⚠️ Which applicants are most likely to default?
* 💰 How do income and loan amount affect default risk?
* 📉 Which loan grades and purposes carry the highest risk?
* 🏦 How does credit history influence repayment behavior?
* 📈 What interest rates and loan-to-income ratios signal danger?
* 🎯 How can lenders score applicants as Low, Medium, or High risk?

---

## 🧰 Tech Stack

| Technology              | Purpose                                            |
| ----------------------- | -------------------------------------------------- |
| 🐍 **Python**           | Data cleaning, preprocessing & feature engineering |
| 🐼 **Pandas**           | Data manipulation and analysis                     |
| 🔢 **NumPy**            | Numerical operations                               |
| 🤖 **Scikit-learn**     | Model training and evaluation                      |
| 📈 **Matplotlib / Seaborn** | Data visualization                             |
| 🌐 **Streamlit**        | Interactive risk prediction app                    |
| 📓 **Jupyter Notebook** | Exploratory data analysis                          |

---

## 📊 Dataset Overview

The dataset contains **[N] loan records** and **[M] features** covering applicant demographics, financial profile, and loan details.

### Key Data Categories

**👤 Applicant Information**

* Age
* Income
* Employment length
* Home ownership

**🏦 Loan Information**

* Loan amount
* Loan purpose
* Loan grade
* Interest rate
* Loan-to-income ratio

**📜 Credit History**

* Previous default on file
* Credit history length

### Dataset Statistics

| Metric             |                 Value |
| ------------------ | --------------------: |
| Total Records      |               **[N]** |
| Total Features     |               **[M]** |
| Missing Values     |               **[X]** |
| Default Rate       |            **[X %]**  |
| Target Variable    |  `[target_column]`    |

---

## 🔄 Project Workflow

```text
                 ┌─────────────────────┐
                 │   Raw Loan Data     │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Data Cleaning &     │
                 │ Preprocessing       │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Feature Engineering │
                 │ & EDA               │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Model Training &    │
                 │ Evaluation          │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Streamlit App       │
                 │ Risk Scoring        │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Business Insights   │
                 │ & Recommendations   │
                 └─────────────────────┘
```

---

## 🧹 1. Data Cleaning & Preparation

### Key Steps

* Loaded and explored the dataset using **Pandas**
* Checked data types and statistical summaries
* Handled missing values and removed unrealistic outliers
* Encoded categorical variables
* Created derived features such as **loan-to-income ratio**
* Split data into stratified train and test sets
* Addressed class imbalance using `[class weights / SMOTE]`

---

## 🔍 2. Exploratory Data Analysis

### Analysis Areas

📊 **Default Analysis**

* Default rate by loan grade
* Default rate by loan purpose
* Default rate by home ownership

💰 **Financial Analysis**

* Income vs default risk
* Loan amount and interest rate patterns
* Loan-to-income ratio distribution

📜 **Credit History Analysis**

* Impact of previous defaults
* Credit history length vs repayment

---

## 🤖 3. Machine Learning Models

Multiple models were trained and compared on the same data split.

| Model               | Accuracy | Precision | Recall | F1  | ROC-AUC |
| ------------------- | :------: | :-------: | :----: | :-: | :-----: |
| Logistic Regression |    –     |     –     |   –    |  –  |    –    |
| Random Forest       |    –     |     –     |   –    |  –  |    –    |
| XGBoost             |    –     |     –     |   –    |  –  |    –    |

> 🏆 **Best model:** `[model name]` with ROC-AUC of **[value]**.

---

## 🌐 4. Streamlit App

The final model is deployed as an interactive app for real-time risk scoring.

### App Highlights

* 🔮 **Predict tab:** enter an applicant profile and get a default probability gauge and a Low / Medium / High risk badge
* 📁 **Batch Prediction tab:** upload a CSV, score every applicant, and download the results
* ℹ️ **About the Model tab:** feature importance and default-rate breakdowns

---

## 💡 Key Insights

### 📉 Loan Grade

Lower loan grades show noticeably higher default rates, making grade a strong risk signal.

### 💰 Loan-to-Income Ratio

Applicants with a high loan-to-income ratio are significantly more likely to default.

### 📈 Interest Rate

Higher interest rates are associated with riskier borrowers and higher default probability.

### 📜 Credit History

Applicants with a previous default on file carry a markedly higher risk.

> ✏️ *Replace these with the exact findings and numbers from your analysis.*

---

## 📌 Business Recommendations

### 1. 🎯 Risk-Based Approval

Use the model's default probability to approve, review, or decline applications.

### 2. 💳 Risk-Based Pricing

Set interest rates in line with each applicant's predicted risk level.

### 3. 🚨 Flag High-Risk Segments

Apply stricter checks to high-risk grades, purposes, and high loan-to-income applicants.

### 4. 📊 Monitor the Portfolio

Track predicted risk across the loan book and retrain the model regularly.

---

## 📂 Project Structure

```text
credit-risk/
│
├── 📓 credit_risk.ipynb
│   └── Data cleaning, EDA & model training
│
├── 🌐 app.py
│   └── Streamlit risk prediction app
│
├── 🤖 model.pkl
│   └── Trained model (+ feature list)
│
├── 🔤 encoders.pkl
│   └── Encoders for categorical columns
│
├── 📊 data.csv
│   └── Credit risk dataset
│
├── 📄 requirements.txt
│   └── Python dependencies
│
└── 📘 README.md
    └── Project documentation
```

> ✏️ *Update this tree to match the actual files in your repo.*

---

## 🚀 How to Run the Project

### Step 1 — Clone the Repository

```bash
git clone https://github.com/darksoul0-0/credit-risk.git
cd credit-risk
```

### Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Explore the Notebook

```text
credit_risk.ipynb
```

### Step 4 — Launch the App

```bash
streamlit run app.py
```

Your browser will open at `http://localhost:8501`.

---

## 📊 Skills Demonstrated

* 🐍 Python for Data Analytics
* 🐼 Pandas & 🔢 NumPy
* 🧹 Data Cleaning
* 🔧 Feature Engineering
* 🔍 Exploratory Data Analysis
* 🤖 Machine Learning (Classification)
* ⚖️ Handling Imbalanced Data
* 📏 Model Evaluation (ROC-AUC, Precision, Recall)
* 🌐 Streamlit App Development
* 💼 Credit Risk & Business Problem Solving

---

## 🌟 Why This Project Matters

This project goes beyond building a model. It demonstrates an **end-to-end workflow**:

> **Raw Data → Clean Data → Insights → Model → Deployed App → Business Recommendations**

The objective is not just to predict *who might default*, but to show **how a lender can act on that prediction**.

---

## 🔮 Future Improvements

* 🎛️ Hyperparameter tuning (Optuna / GridSearchCV)
* 🧠 Model explainability with SHAP
* ⚖️ Probability calibration and cost-based threshold tuning
* 📈 Credit scorecard development
* ⚡ Automated retraining pipeline

---

## 👨‍💻 Author

**Darksoul0-0**

📊 Data Analytics | Machine Learning | Python

If you found this project useful, consider giving the repository a ⭐.

---

## 📜 License

This project is intended for educational and portfolio purposes.
