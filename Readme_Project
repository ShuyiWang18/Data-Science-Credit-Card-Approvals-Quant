Credit Card Approval Prediction Using Machine Learning

Overview
This project implements a complete pipeline for predicting credit card approval decisions by combining data preprocessing, feature engineering, imbalance handling, and multiple machine learning algorithms. It demonstrates how to:
Load and clean two datasets—applicant profiles and repayment history ​
Engineer features using Weight of Evidence (WoE) / Information Value (IV) and category binning ​
Balance the target classes with SMOTE ​
Train & evaluate various models (Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, SVM) ​

Files:
application_record 2.csv
Applicant demographic and application data.

credit_record 2.csv
Monthly repayment records, used to derive the binary approval target.

Requirments:
Python 3.x
pandas
numpy
matplotlib
seaborn
imbalanced-learn
scikit-learn
xgboost
lightgbm

Usage
Place application_record 2.csv and credit_record 2.csv in the same directory as the notebook.

Open the HTML in JupyterLab / Jupyter Notebook or convert to .ipynb.

Execute cells in order:

Data Loading & Renaming
Reads the two CSVs and renames columns for consistency.

Preprocessing
Merge datasets, derive dep_value (“Yes”/“No”) and binary target.
Drop missing / 'NULL' values ​

Feature Engineering
Compute WoE and IV for each categorical variable to assess predictive power.
Bin continuous income with qcut into “low”, “medium”, “high” groups.
One‐hot encode categories via custom convert_dummy() ​

Imbalance Handling
Apply SMOTE to oversample the minority class:
python
X_balance, Y_balance = SMOTE().fit_resample(X, Y)

Key Code:
%matplotlib inline
%config InlineBackend.figure_format = 'svg'

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import itertools

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

Results
Confusion Matrix and Accuracy comparisons across models.
Feature importance plots for tree‐based models.
Insights on which variables (e.g., income group, occupation type) most influence approval.

