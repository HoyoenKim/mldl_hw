import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# (1)
auto_data = pd.read_csv("./Auto.csv", header=0, na_values="?")
auto_data = auto_data.dropna()
#print(auto_data.shape)
#print(auto_data.describe())

# (a)
# quantitative: mpg, cylinders, displacement, horsepower, weight, acceleration, year
# qualitative: name, origin

# (b)
# apply the range function to the first seven columns of auto_data
print('b')
print(auto_data.iloc[:, 0:7].agg([min, max]))

# (c)
print('c')
print(auto_data.iloc[:, 0:7].mean())
print(auto_data.iloc[:, 0:7].std())

# (d)
print('d')
new_auto_data = auto_data.drop(auto_data.index[9:85])
#print(new_auto_data.shape == (auto_data.shape[0] - 76, auto_data.shape[1]))
#print((new_auto_data.iloc[8] == auto_data.iloc[8]).all())
#print((new_auto_data.iloc[9] == auto_data.iloc[85]).all())

print(new_auto_data.iloc[:, 0:7].agg([min, max]))
print(new_auto_data.iloc[:, 0:7].mean())
print(new_auto_data.iloc[:, 0:7].std())

# (e)
pd.plotting.scatter_matrix(auto_data, figsize=(12, 12))
plt.savefig("scatter_matrix.png")
plt.clf()

plt.scatter(auto_data['mpg'], auto_data['weight'])
plt.xlabel('mpg')
plt.ylabel('weight')
plt.savefig("mpg_vs_weight.png")
plt.clf()

plt.scatter(auto_data['mpg'], auto_data['cylinders'])
plt.xlabel('mpg')
plt.ylabel('cylinders')
plt.savefig("mpg_vs_cylinders.png")
plt.clf()

plt.scatter(auto_data['mpg'], auto_data['year'])
plt.xlabel('mpg')
plt.ylabel('year')
plt.savefig("mpg_vs_year.png")
plt.clf()

# (f)
pd.plotting.scatter_matrix(auto_data, figsize=(12, 12))
plt.savefig("scatter_matrix_f.png")
plt.clf()