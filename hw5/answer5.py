import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

auto_data = pd.read_csv('./Auto.csv')
auto_columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year', 'origin']
for column in auto_columns:
    auto_data[column] = pd.to_numeric(auto_data[column], errors='coerce')
    auto_data[column].fillna(auto_data[column].median(), inplace=True)
    
mpg_median = auto_data['mpg'].median()
auto_data['mpg_binary'] = np.where(auto_data['mpg'] > mpg_median, 1, 0)

X = auto_data.drop(['mpg', 'mpg_binary', 'name'], axis=1)
X = StandardScaler().fit_transform(X)
y = auto_data['mpg_binary']

C_values = [0.01, 0.1, 1, 10, 100]
gamma_values = [0.01, 0.1, 1, 10, 100]
degree_values = [2, 3, 4, 5]

# (a)
print('Linear Kernel')
for C in C_values:
    svc = SVC(C = C, kernel = 'linear')
    scores = cross_val_score(svc, X, y, cv=5)
    print(f'C = {C}: Cross-Validation Accuracy = {np.mean(scores)}')
print()

# (b)
print("Radial Basis Function Kernel")
for C in C_values:
    for gamma in gamma_values:
        svc = SVC(C = C, kernel = 'rbf', gamma = gamma)
        scores = cross_val_score(svc, X, y, cv=5)
        print(f'C = {C}, gamma = {gamma}: Cross-Validation Accuracy = {np.mean(scores)}')
print()

print("Polynomial Kernel")
for C in C_values:
    for degree in degree_values:
        svc = SVC(C = C, kernel = 'poly', degree = degree)
        scores = cross_val_score(svc, X, y, cv=5)
        print(f'C={C}, degree={degree}: Cross-Validation Accuracy = {np.mean(scores)}')

print()