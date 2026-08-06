import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression as sklearn_LR
from sklearn.metrics import mean_squared_error, r2_score

from algorithms.supervised.linear_regression import LinearRegression as custom_LR

housing_data = fetch_california_housing(as_frame=True)

X = housing_data.data
y = housing_data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

custom_model = custom_LR()
custom_model.fit(X_train_scaled, y_train)
y_pred = custom_model.predict(X_test_scaled)

print(f'Mean squared error of the custom model: {mean_squared_error(y_test, y_pred)}')
print(f'R2 Score of the custom model: {r2_score(y_test, y_pred)}')

sklearn_model = sklearn_LR()
sklearn_model.fit(X_train_scaled, y_train)
y_pred = sklearn_model.predict(X_test_scaled)

print(f'Mean squared error of the sklearn model: {mean_squared_error(y_test, y_pred)}')
print(f'R2 Score of the sklearn model: {r2_score(y_test, y_pred)}')