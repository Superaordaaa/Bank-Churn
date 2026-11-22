# Bank Customer Churn — EDA & Classification

Team final project ("Boba Warriors") from a Rakamin Academy data science bootcamp (2023). The goal is to find out which customers are likely to leave a bank, and why.

**Dataset:** the public `Churn_Modelling.csv` (bank customer data; target `Exited`).

## What's inside (`final_project_boba_warriors.py`, exported from Google Colab)
1. **EDA:** distributions, boxplots and violin plots for outliers, plus churn rate by country, tenure, and credit score.
2. **Data cleansing:** dropped unused columns, handled duplicates, and removed outliers with z-score < 3.
3. **Feature engineering:** added a low-credit-score flag (< 400), label and one-hot encoding, standardization, and class-imbalance handling.
4. **Modeling:** Logistic Regression, KNN, and Decision Tree, tuned with RandomizedSearchCV/GridSearchCV and evaluated with accuracy, precision, recall, F1, ROC-AUC, 5-fold cross-validation, and learning curves.

## Run
Open the script in Google Colab (it uses `google.colab.files.upload()`), then upload `Churn_Modelling.csv` when prompted.

Notes and comments in the code are in Indonesian.
