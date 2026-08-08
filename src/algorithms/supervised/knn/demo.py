import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier as sklearn_knn_clf
from sklearn.neighbors import KNeighborsRegressor as sklearn_knn_reg
from sklearn.metrics import roc_auc_score, f1_score

from algorithms.supervised.knn import KNeighborsClassifier
from algorithms.supervised.knn import KNeighborsRegressor

np.random.seed(42)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cancer_data = load_breast_cancer()
X, y = cancer_data.data, cancer_data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

custom_knn_clf = KNeighborsClassifier(k=5)
custom_knn_clf.fit(X_train_scaled, y_train)
y_pred_prob = custom_knn_clf.predict_proba(X_test_scaled)[:, 1]
y_pred = custom_knn_clf.predict(X_test_scaled)
print(f'Custom KNN classifier ROC AUC score: {roc_auc_score(y_test, y_pred_prob):.3f} | F1 score: {f1_score(y_test, y_pred):.3f}')

scores = []
for train_idx, val_idx in cv.split(X, y):
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]

    scaler = StandardScaler()
    X_tr_fold_scaled = scaler.fit_transform(X_train_fold)
    X_val_fold_scaled = scaler.transform(X_val_fold)

    model = KNeighborsClassifier(k=5)
    model.fit(X_tr_fold_scaled, y_train_fold)
    y_pred = model.predict_proba(X_val_fold_scaled)[:, 1]
    scores.append(roc_auc_score(y_val_fold, y_pred))

print(f"Mean custom KNN ROC AUC: {np.mean(scores):.3f} | Std: {np.std(scores):.3f}")

sklearn_knn_clf = sklearn_knn_clf(n_neighbors=5)
sklearn_knn_clf.fit(X_train_scaled, y_train)
y_pred_prob = sklearn_knn_clf.predict_proba(X_test_scaled)[:, 1]
y_pred = sklearn_knn_clf.predict(X_test_scaled)
print(f'Sklearn KNN classifier ROC AUC score: {roc_auc_score(y_test, y_pred_prob):.3f} | F1 score: {f1_score(y_test, y_pred):.3f}')

scores = cross_val_score(sklearn_knn_clf, X_train_scaled, y_train, cv=cv, scoring='roc_auc')
print(f"Mean sklearn KNN ROC AUC: {np.mean(scores):.3f} | Std: {np.std(scores):.3f}")

