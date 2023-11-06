import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

# (a)
def polynomial_regression(degree, X, y):
    X = X.to_numpy() 
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    mse = -np.mean(cross_val_score(model, X.reshape(-1, 1), y, cv=10, scoring='neg_mean_squared_error'))
    return mse

def a(wage):
    X = wage['age']
    y = wage['wage']

    degrees = range(1, 11)
    mse_values = [polynomial_regression(d, X, y) for d in degrees]
    best_degree = degrees[np.argmin(mse_values)]

    best_model = make_pipeline(PolynomialFeatures(best_degree), LinearRegression())
    best_model.fit(X.values.reshape(-1, 1), y)  # Modify this line

    X_range = np.arange(X.min(), X.max(), 0.1).reshape(-1, 1)
    y_pred = best_model.predict(X_range)

    plt.figure()
    plt.scatter(X, y, s=5)
    plt.plot(X_range, y_pred, color='green', linewidth=2)
    plt.title(f'Polynomial Regression (Degree {best_degree})')
    plt.xlabel('Age')
    plt.ylabel('Wage')

    plt.savefig('polynomial_regression.png')

# (b)
def step_function(num_cuts, X, y):
    cut_points = np.percentile(X, np.linspace(0, 100, num_cuts))
    X_cut = np.digitize(X, cut_points)

    mse_values = []
    for fold in range(10):
        start = fold * len(X) // 10
        end = (fold + 1) * len(X) // 10

        X_train = np.concatenate([X_cut[:start], X_cut[end:]])
        X_valid = X_cut[start:end]
        y_train = np.concatenate([y[:start], y[end:]])
        y_valid = y[start:end]

        model = make_pipeline(LinearRegression())
        model.fit(X_train.reshape(-1, 1), y_train)

        y_pred = model.predict(X_valid.reshape(-1, 1))
        mse_values.append(np.mean((y_valid - y_pred) ** 2))

    return np.mean(mse_values)

def b(wage):
    X = wage['age']
    y = wage['wage']

    num_cuts_values = range(2, 11)
    mse_values = [step_function(num_cuts, X, y) for num_cuts in num_cuts_values]

    best_num_cuts = num_cuts_values[np.argmin(mse_values)]

    cut_points = np.percentile(X, np.linspace(0, 100, best_num_cuts))
    X_cut = np.digitize(X, cut_points)

    model = make_pipeline(LinearRegression())
    model.fit(X_cut.reshape(-1, 1), y)

    X_range = np.arange(X.min(), X.max(), 0.1).reshape(-1, 1)
    y_pred = model.predict(X_range)

    plt.figure()
    plt.scatter(X, y, s=5)
    plt.plot(X_range, y_pred, color='green', linewidth=2)
    plt.title(f'Step Function (Cuts {best_num_cuts})')
    plt.xlabel('Age')
    plt.ylabel('Wage')

    plt.savefig('step_function.png')

wage = pd.read_csv("./Wage.csv")
a(wage)
b(wage)