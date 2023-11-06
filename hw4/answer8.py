import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge

# (a)
def a():
    n_trees = 1000
    shrinkage_values = [0.001, 0.01, 0.1, 0.5, 1]
    train_mse_values = []
    test_mse_values = []
    for shrinkage in shrinkage_values:
        model = GradientBoostingRegressor(n_estimators=n_trees, learning_rate=shrinkage, random_state=42)
        model.fit(X_train, y_train)
    
        y_train_pred = model.predict(X_train)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_mse_values.append(train_mse)

        y_test_pred = model.predict(X_test)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_mse_values.append(test_mse)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(shrinkage_values, train_mse_values, marker='o')
    plt.xlabel('Shrinkage')
    plt.ylabel('Training Set MSE')
    plt.title('Training Set MSE vs. Shrinkage')

    plt.subplot(1, 2, 2)
    plt.plot(shrinkage_values, test_mse_values, marker='o')
    plt.xlabel('Shrinkage')
    plt.ylabel('Test Set MSE')
    plt.title('Test Set MSE vs. Shrinkage')
    plt.tight_layout()
    plt.savefig('Training and Test MSE Shrinkage')

# (b) Compare test MSE with other regression approaches (e.g., linear regression)
def b():
    # Boosting
    n_trees = 1000
    shrinkage = 0.1
    boosting_model = GradientBoostingRegressor(n_estimators=n_trees, learning_rate=shrinkage, random_state=42)
    boosting_model.fit(X_train, y_train)
    y_test_boosting_pred = boosting_model.predict(X_test)
    test_mse_boosting = mean_squared_error(y_test, y_test_boosting_pred)

    # Linear Regression
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_test_linear_pred = linear_model.predict(X_test)
    test_mse_linear = mean_squared_error(y_test, y_test_linear_pred)

    # Ridge Regression
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train, y_train)
    y_test_ridge_pred = ridge_model.predict(X_test)
    test_mse_ridge = mean_squared_error(y_test, y_test_ridge_pred)

    print(f"Test MSE - Boosting: {test_mse_boosting}")
    print(f"Test MSE - Linear Regression: {test_mse_linear}")
    print(f"Test MSE - Ridge Regression: {test_mse_ridge}")

# (c)
def c():
    n_trees = 1000
    model = GradientBoostingRegressor(n_estimators=n_trees, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    feature_importance = model.feature_importances_
    feature_names = X_train.columns
    importance_dict = dict(zip(feature_names, feature_importance))

    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    print("Feature Importance:")
    for feature, importance in sorted_importance:
        print(f"{feature}: {importance}")

hitters = pd.read_csv("Hitters.csv")
hitters = hitters.dropna(subset=['Salary'])
hitters['Salary'] = np.log(hitters['Salary'])

X = hitters.drop(columns=['Salary', 'Unnamed: 0', 'League', 'Division', 'NewLeague'])
y = hitters['Salary']

X_train, X_test = X.iloc[:200], X.iloc[200:]
y_train, y_test = y.iloc[:200], y.iloc[200:]

a()
b()
c()