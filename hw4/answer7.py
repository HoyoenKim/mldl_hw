import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor

# (a)
def preprocess_data(carseats):
    carseats_encoded = pd.get_dummies(carseats, columns=['ShelveLoc', 'Urban', 'US'], drop_first=True)
    return carseats_encoded

def a(carseats):
    columns = selected_features = ["Sales", "CompPrice", "Income", "Advertising", "Population", "Price", "ShelveLoc", "Age", "Education", "Urban", "US"]
    carseats_encoded = preprocess_data(carseats[columns])
    X = carseats_encoded.drop(['Sales'], axis=1)
    y = carseats_encoded['Sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    reg_tree = DecisionTreeRegressor()
    reg_tree.fit(X_train, y_train)

    plot_tree(reg_tree, filled=True, feature_names=X.columns)
    plt.savefig('plot_tree.png')

    y_pred = reg_tree.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"Regression Tree Test MSE: {mse}")

# (b)
def b(carseats):
    carseats_encoded = preprocess_data(carseats)
    X = carseats_encoded.drop(['Sales'], axis=1)
    y = carseats_encoded['Sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    max_depth_values = list(range(1, 21))
    mse_values = []

    for max_depth in max_depth_values:
        reg_tree = DecisionTreeRegressor(max_depth=max_depth)
        reg_tree.fit(X_train, y_train)
        y_pred = reg_tree.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mse_values.append(mse)

    optimal_max_depth = max_depth_values[np.argmin(mse_values)]
    min_mse = min(mse_values)
    
    print(f"Optimal Max Depth: {optimal_max_depth}")
    print(f"Optimal Regression Tree Test MSE: {min_mse}")

    reg_tree_optimal = DecisionTreeRegressor(max_depth=optimal_max_depth)
    reg_tree_optimal.fit(X_train, y_train)
    
    plot_tree(reg_tree_optimal, filled=True, feature_names=X.columns)
    plt.savefig('plot_tree_optimal.png')

# (c)
def c(carseats, optimal_max_depth):
    carseats_encoded = preprocess_data(carseats)
    X = carseats_encoded.drop(['Sales'], axis=1)
    y = carseats_encoded['Sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    bagging = BaggingRegressor(base_estimator=DecisionTreeRegressor(max_depth=optimal_max_depth),
                              n_estimators=100, random_state=42)
    bagging.fit(X_train, y_train)

    y_pred = bagging.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    print(f"Bagging Test MSE: {mse}")

# (d)
def d(carseats):
    carseats_encoded = preprocess_data(carseats)
    X = carseats_encoded.drop(['Sales'], axis=1)
    y = carseats_encoded['Sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=100, max_features='sqrt', random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    print(f"Random Forest Test MSE: {mse}")

    feature_importance = rf.feature_importances_
    feature_names = X.columns
    importance_dict = dict(zip(feature_names, feature_importance))

    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    for feature, importance in sorted_importance:
        print(f"{feature}: {importance}")

carseats = pd.read_csv("./Carseats.csv")
a(carseats)
optimal_max_depth = b(carseats)
c(carseats, optimal_max_depth)
d(carseats)