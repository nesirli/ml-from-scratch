import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as sklearn_LogReg
from sklearn.metrics import roc_auc_score, f1_score

from algorithms.supervised.logistic_regression import LogisticRegression as custom_LogReg

np.random.seed(42)

cancer_data = load_breast_cancer(as_frame=True)

X = cancer_data.data
y = cancer_data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

custom_model = custom_LogReg()
custom_model.fit(X_train_scaled, y_train)
y_pred_proba = custom_model.predict_proba(X_test_scaled)
y_pred = custom_model.predict(X_test_scaled)

print(f'ROC AUC score of the custom model: {roc_auc_score(y_test, y_pred_proba):.3f}')
print(f'F1 Score of the custom model: {f1_score(y_test, y_pred):.3f}')

sklearn_model = sklearn_LogReg()
sklearn_model.fit(X_train_scaled, y_train)
y_pred_proba = sklearn_model.predict_proba(X_test_scaled)[:, 1]
y_pred = sklearn_model.predict(X_test_scaled)

print(f'ROC AUC score of the sklearn model: {roc_auc_score(y_test, y_pred_proba):.3f}')
print(f'F1 Score of the sklearn model: {f1_score(y_test, y_pred):.3f}')