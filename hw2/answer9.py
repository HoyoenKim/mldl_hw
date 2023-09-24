import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv("Smarket.csv")

#(a)
summary = data.describe()
print(summary)

sns.pairplot(data)
plt.savefig("pairplot.png")
plt.clf()

correlation_data = data[['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume']]
correlation_matrix = correlation_data.corr()

plt.figure()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.clf()

# (b)
x = data[['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume']]
y = data['Direction'].map({'Up': 1, 'Down': 0})

x = sm.add_constant(x)
model = sm.Logit(y, x).fit()
print(model.summary())

# (c)
predicted = (model.predict(x) > 0.5).astype(int)
confusion = confusion_matrix(y, predicted)
accuracy = accuracy_score(y, predicted)

print("Confusion Matrix:")
print(confusion)
print("Accuracy:", accuracy)

# (d)
train_data = data[data['Year'] < 2005]
test_data = data[data['Year'] >= 2005]

X_train = train_data[['Lag2']]
y_train = train_data['Direction'].map({'Up': 1, 'Down': 0})

X_test = test_data[['Lag2']]
y_test = test_data['Direction'].map({'Up': 1, 'Down': 0})

X_train = sm.add_constant(X_train)
model = sm.Logit(y_train, X_train).fit()

X_test = sm.add_constant(X_test)
predicted = (model.predict(X_test) > 0.5).astype(int)
confusion = confusion_matrix(y_test, predicted)
accuracy = accuracy_score(y_test, predicted)

print("Confusion Matrix:")
print(confusion)
print("Accuracy:", accuracy)

# (e)
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

predicted = lda.predict(X_test)
confusion = confusion_matrix(y_test, predicted)
accuracy = accuracy_score(y_test, predicted)

print("Confusion Matrix (LDA):")
print(confusion)
print("Accuracy (LDA):", accuracy)

#(f)
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)

predicted = qda.predict(X_test)
confusion = confusion_matrix(y_test, predicted)
accuracy = accuracy_score(y_test, predicted)

print("Confusion Matrix (QDA):")
print(confusion)
print("Accuracy (QDA):", accuracy)

#(g)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

predicted = knn.predict(X_test)
confusion = confusion_matrix(y_test, predicted)
accuracy = accuracy_score(y_test, predicted)

print("Confusion Matrix (KNN with K=1):")
print(confusion)
print("Accuracy (KNN with K=1):", accuracy)

#(h)
nb = GaussianNB()
nb.fit(X_train, y_train)

predicted = nb.predict(X_test)
confusion = confusion_matrix(y_test, predicted)
accuracy = accuracy_score(y_test, predicted)

print("Confusion Matrix (Naive Bayes):")
print(confusion)
print("Accuracy (Naive Bayes):", accuracy)