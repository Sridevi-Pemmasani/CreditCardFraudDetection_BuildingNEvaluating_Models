# 🔐 Credit Card Fraud Prediction Using Multiple Machine Learning Models and Evaluation of Results

---

## 👩‍💻 Team Members
| Name                
|---------------------
| Sridevi Pemmasani      
| Catherine Calantoc  
| Jayasri Katragadda  
| Arya Kaippully 

---

## 📌 Project Overview

Credit card fraud is a critical issue affecting both financial institutions and consumers. The goal of this project is to develop a **robust fraud detection model** using various machine learning techniques to identify fraudulent transactions with high precision and recall, while minimizing false positives.

---

## 🎯 Objectives

- Enhance fraud detection accuracy while reducing false positives.
- Address class imbalance using transformation and sampling techniques.
- Compare model performance using metrics like **Accuracy**, **Precision**, **Recall**, and **AUC-ROC**.
- Deliver a **scalable and interpretable** solution suitable for real-world application.

---

## 🧠 Research Questions

1. Which ML model best detects fraudulent transactions?
2. How do transformation techniques improve model performance?
3. What impact does feature selection have on fraud detection?
4. Can the model be adapted for real-time fraud detection?

---

## 📊 Dataset Description

- **Source**: Kaggle – [Credit Card Fraud Detection](https://www.kaggle.com/datasets)
- **Size**: ~14,383 transactions
- **Target**: `is_fraud` (1 = Fraud, 0 = Legitimate)
- **Key Features**: Transaction amount, time, category, location, flags (POS/Net), demographic info.

---

## 🔍 Methodology

### 1. Data Preprocessing
- Null values removed
- Duplicates and outliers handled (IQR method)
- One-hot encoding for categorical data
- Feature importance & multicollinearity check (VIF)

### 2. Data Transformation
| Transformation | Purpose |
|----------------|---------|
| Log            | Reduce skew/outliers |
| Box-Cox        | Normalize non-Gaussian data |
| Square Root    | Stabilize variance |
| StandardScaler | Normalize features |

### 3. Machine Learning Models Used
- ✅ **Logistic Regression**
- ✅ **Decision Tree**
- ✅ **Random Forest**
- ✅ **GaussianNB**
- ✅ **LDA/QDA**
- ✅ **KNN**
- 🔜 (Planned): **XGBoost**, **Autoencoders (Unsupervised)**

---

## 📈 Model Evaluation Metrics

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **AUC-ROC**
- **Confusion Matrix**
- **Residual and Q-Q Plots**
- **Breusch-Pagan Test** (Homoscedasticity)
- **Variance Inflation Factor (VIF)**

---

## 🏆 Model Performance Summary

| Model               | Accuracy | Precision | Recall | AUC-ROC |
|---------------------|----------|-----------|--------|---------|
| Logistic Regression | 80.5%    | Moderate  | Low    | -       |
| Decision Tree       | 94.5%    | 92%       | 96%    | High    |
| GaussianNB          | 72.0%    | 68.9%     | 74.3%  | -       |
| KNN, LDA, QDA       | Compared in report |

✅ **Best performer:** Decision Tree Classifier (after tuning with `max_depth=7`, `min_samples_split=20`)

---

## 📌 Key Insights

- Fraudulent transactions are concentrated in **online shopping**, **grocery POS**, and **miscellaneous net** categories.
- Fraud transactions are **on average 7x higher** in amount than legitimate ones.
- Decision Tree & ensemble models outperform traditional logistic regression for fraud detection.

---

## 💼 Business Impact

- Reduces financial loss from unauthorized transactions
- Increases customer trust with fewer false alarms
- Enables real-time fraud monitoring with automated alerts
- Improves compliance with interpretable models

---

## 🧪 Future Work

- Implement **SMOTE** for better class balance
- Deploy **Random Forest & XGBoost**
- Explore **SHAP/LIME** for explainable AI
- Consider **Autoencoders** or **Isolation Forests** for anomaly detection
- Real-time fraud detection with **Kafka + Spark** pipeline (prototype)

---

## 📂 Repository Structure

📁 Credit-Card-Fraud-Detection/
│
├── 📄 README.md # Project overview (this file)
├── 📄 Credit Card Fraud Detection_Model Building.ipynb # Main Jupyter notebook
├── 📄 report.pdf # Final academic report
├── 📊 data/
│ └── creditcard.csv # Cleaned dataset (or link if sensitive)
└── 📈 images/
└── *.png # Plots, charts, confusion matrices