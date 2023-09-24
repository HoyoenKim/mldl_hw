import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

carseats = pd.read_csv("Carseats.csv")
carseats['Urban'] = carseats['Urban'].map({'Yes': 1, 'No': 0})
carseats['US'] = carseats['US'].map({'Yes': 1, 'No': 0})

# (a), (b), (c), (d)
predictors = ['Price', 'Urban', 'US']
x = sm.add_constant(carseats[predictors])
y = carseats['Sales']
model = sm.OLS(y, x).fit()
print(model.summary())

# (e) (f)
predictors_2 = ['Price', 'US']
x_2 = sm.add_constant(carseats[predictors_2])
model_2 = sm.OLS(y, x_2).fit()
print(model_2.summary())


# (h)
# Residuals vs. Leverage Plot
from statsmodels.graphics.regressionplots import plot_leverage_resid2
plot_leverage_resid2(model_2)
plt.title('Residuals vs Leverage')
plt.savefig('residuals vs leverage.png')
plt.clf()