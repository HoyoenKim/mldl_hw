import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

# (a)
rng = np.random.default_rng(10)
x1 = rng.uniform(0, 1, size=100)
x2 = 0.5 * x1 + rng.normal(size=100) / 10
y = 2 + 2 * x1 + 0.3 * x2 + rng.normal(size=100)

# (b)
correlation = np.corrcoef(x1, x2)[0, 1]
print("Correlation between x1 and x2:", correlation)
plt.scatter(x1, x2)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("x1 vs x2")
plt.savefig("x1 vs x2.png")
plt.clf()

# (c)
x = sm.add_constant(np.column_stack((x1, x2)))
model = sm.OLS(y, x).fit()

beta0 = model.params[0]
beta1 = model.params[1]
beta2 = model.params[2]

print("Intercept:", beta0)
print("Coefficient for x1:", beta1)
print("Coefficient for x2:", beta2)
print(model.summary())

# (d)
x1_model = sm.add_constant(x1)
model_x1 = sm.OLS(y, x1_model).fit()

beta0_x1 = model_x1.params[0]
beta1_x1 = model_x1.params[1]

print("Intercept:", beta0_x1)
print("Coefficient for x1:", beta1_x1)
print(model_x1.summary())

# (e)
x2_model = sm.add_constant(x2)
model_x2 = sm.OLS(y, x2_model).fit()

beta0_x2 = model_x2.params[0]
beta1_x2 = model_x2.params[1]

print("Intercept:", beta0_x2)
print("Coefficient for x2:", beta1_x2)
print(model_x2.summary())

# (g)
x1 = np.concatenate([x1, [0.1]])
x2 = np.concatenate([x2, [0.8]])
y = np.concatenate([y, [6]])

correlation_new = np.corrcoef(x1, x2)[0, 1]
print("Correlation between x1_new and x2_new:", correlation_new)
plt.scatter(x1, x2)
plt.xlabel("x1_new")
plt.ylabel("x2_new")
plt.title("x1_new vs x2_new")
plt.savefig("x1_new vs x2_new.png")
plt.clf()

# Fit the models with the new data
x_new = sm.add_constant(np.column_stack((x1, x2)))
model_new = sm.OLS(y, x_new).fit()

x1_model_new = sm.add_constant(x1)
model_x1_new = sm.OLS(y, x1_model_new).fit()

x2_model_new = sm.add_constant(x2)
model_x2_new = sm.OLS(y, x2_model_new).fit()

print("Model with both x1 and x2:")
print(model_new.summary())

print("Model with only x1:")
print(model_x1_new.summary())

print("Model with only x2:")
print(model_x2_new.summary())

# Residuals vs. Leverage Plot
from statsmodels.graphics.regressionplots import plot_leverage_resid2
plot_leverage_resid2(model_new)
plt.title('Residuals vs Leverage (model_new)')
plt.savefig('residuals vs leverage model_new.png')
plt.clf()

plot_leverage_resid2(model_x1_new)
plt.title('Residuals vs Leverage (model_x1_new)')
plt.savefig('residuals vs leverage model_x1_new.png')
plt.clf()

plot_leverage_resid2(model_x2_new)
plt.title('Residuals vs Leverage (model_x2_new)')
plt.savefig('residuals vs leverage model_x2_new.png')
plt.clf()

