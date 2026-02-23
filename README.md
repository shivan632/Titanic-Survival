# 🚢 Titanic Survival Prediction using Machine Learning

> Internship Project – Data Science & Classification Modeling  
> Role: Data Scientist Intern  
> Author: Shivan Mishra  

---

## 📌 Project Overview

This project focuses on predicting passenger survival on the Titanic using machine learning classification models.

The objective is to analyze passenger data, perform preprocessing and feature engineering, train multiple models, and evaluate their performance to determine survival probability.

The project demonstrates end-to-end implementation of a real-world supervised learning pipeline.

---

## 🎯 Business Objective

The goal of this project is to:

- Analyze historical passenger data
- Identify key factors affecting survival
- Build predictive classification models
- Evaluate model performance using industry-standard metrics
- Extract important features influencing survival

This type of predictive modeling is widely used in risk assessment and decision analytics.

---

## 📂 Dataset Description

The dataset contains passenger-level information including:

| Feature | Description |
|----------|------------|
| PassengerId | Unique passenger identifier |
| Pclass | Passenger class (1st, 2nd, 3rd) |
| Name | Passenger name |
| Sex | Gender |
| Age | Age of passenger |
| SibSp | Number of siblings/spouses aboard |
| Parch | Number of parents/children aboard |
| Ticket | Ticket number |
| Fare | Ticket fare |
| Cabin | Cabin number |
| Embarked | Port of embarkation |
| Survived | Target variable (0 = No, 1 = Yes) |

---

## 🛠️ Tools & Technologies Used

- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- Logistic Regression  
- Decision Tree  
- Random Forest  
- Model Evaluation Metrics  

---

## 🔍 Project Workflow

### 1️⃣ Data Cleaning

- Handled missing values in Age and Embarked
- Removed or treated irrelevant columns
- Ensured dataset consistency

---

### 2️⃣ Feature Engineering & Encoding

- Encoded categorical variables (Sex, Pclass, Embarked)
- Converted text categories into numerical format
- Prepared dataset for machine learning models

---

### 3️⃣ Model Training

Three classification models were implemented:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

Each model was trained on the processed dataset to predict survival.

---

### 4️⃣ Model Evaluation

Models were evaluated using:

- Accuracy
- Precision
- Recall
- ROC-AUC Score

These metrics help assess classification performance, especially in imbalanced scenarios.

---

### 5️⃣ Feature Importance Analysis

Feature importance analysis revealed that:

- Gender (Sex)
- Passenger Class (Pclass)
- Fare

were among the most significant predictors of survival.

This aligns with historical insights from the Titanic disaster.

---

## 📊 Results & Insights

- Random Forest achieved strong predictive performance.
- Gender played a major role in survival probability.
- Higher-class passengers had better survival chances.
- Fare was positively correlated with survival.

The model successfully identified patterns that influenced passenger survival.

---

## 💼 Business Impact

This project demonstrates how machine learning can:

- Identify critical risk factors
- Support data-driven decision-making
- Improve predictive accuracy
- Extract meaningful insights from structured data

Such classification systems are widely used in insurance, healthcare, and financial risk modeling.

---

## 📁 Project Structure

Titanic-Survival  
│  
├── Titanic_Survival_Prediction.ipynb  
├── titanic.csv  
├── README.md  

---

## 🚀 How to Use the Project

1. Clone the repository from GitHub.  
2. Install required Python libraries.  
3. Open the Jupyter Notebook file.  
4. Run all cells to reproduce preprocessing, modeling, and evaluation.  

---

## 📈 Future Enhancements

- Hyperparameter tuning using GridSearchCV  
- Feature scaling optimization  
- Cross-validation implementation  
- Model comparison visualization  
- Deployment using Flask or Streamlit  

---

## 📌 Conclusion

This internship project successfully demonstrates:

- Data preprocessing and cleaning  
- Feature encoding  
- Supervised classification modeling  
- Model evaluation using multiple metrics  
- Feature importance interpretation  

The final model effectively predicts survival probability and highlights key influencing factors.

---

## 👨‍💻 Author

Shivan Mishra  
Data Scientist Intern  
GitHub: https://github.com/shivan632