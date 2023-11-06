import numpy as np
import pandas as pd

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
medv = target

# (a)
a = medv.mean()
print(a)

# (b)
b = medv.std()/np.sqrt(len(medv))
print(b)

# (c)
n_samples = 1000
means = np.zeros(n_samples)
for i in range(n_samples):
    resampled_data = np.random.choice(medv, size=len(medv), replace=True)
    means[i] = resampled_data.mean()

c = means.std()
print(c)

# (d)
SE = np.std(means)
print(a - 2*SE, a + 2*SE)

# (e)
e = np.median(medv)
print(e)

# (f)
medians = [np.median(np.random.choice(medv, size=len(medv), replace=True)) for _ in range(1000)]
f = np.std(medians)
print(f)

# (g)
g = np.percentile(medv, 10)
print(g)

# (h)
quantiles = [np.percentile(np.random.choice(medv, size=len(medv), replace=True), 10) for _ in range(1000)]
h = np.std(quantiles)
print(h)