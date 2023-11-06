import numpy as np
import pandas as pd
from patsy import dmatrix
from statsmodels.api import OLS
from sklearn.preprocessing import PolynomialFeatures

# (a)
def a(boston):
    degree_of_freedoms = list(range(3, 11))
    rss_values = []
    for degree_of_freedom in degree_of_freedoms:
        formula = f'bs(boston["dis"], df={degree_of_freedom}, include_intercept=False)'
        X = dmatrix(formula, data=boston)
        y = boston['nox']

        model = OLS(y, X).fit()

        rss = ((model.predict(X) - y) ** 2).sum()
        rss_values.append(rss)
    
    for degree_of_freedom, rss in zip(degree_of_freedoms, rss_values):
        print(f"Degree of Freedom: {degree_of_freedom}, RSS: {rss}")

# (b)
def b(boston):
    cv_errors = []
    for degree_of_freedoms in range(3, 11):
        formula = f'bs(boston["dis"], df={degree_of_freedoms}, include_intercept=False)'
        X = dmatrix(formula, data=boston)
        y = boston['nox']
        
        rss_values = []
        for fold in range(10):
            start = fold * len(X) // 10
            end = (fold + 1) * len(X) // 10

            X_train, X_valid = np.vstack((X[:start], X[end:])), X[start:end]
            y_train, y_valid = np.append(y[:start], y[end:]), y[start:end]

            model = OLS(y_train, X_train).fit()

            rss = ((model.predict(X_valid) - y_valid) ** 2).sum()
            rss_values.append(rss)
        
        cv_errors.append(np.mean(rss_values))
    
    best_degree_of_freedom = np.argmin(cv_errors) + 3
    print(f"Best degree of freedom: {best_degree_of_freedom}")

boston = pd.read_csv("./Boston.csv")
a(boston)
b(boston)