## Polynomial Regression Implementation
# %% Importing Libraries
import numpy as np
import pandas as pd
# %% Loading Data-set as a DataFrame
df = pd.read_csv('data.csv')
# %% Looking at the initial data
print(df.head())
# %% Adding a Bias Column for Theta0 of ones
df = pd.concat([pd.Series(1, index=df.index, name='Theta0'), df], axis=1)
print(df.head())
# %% Drop 'Position' Column
df = df.drop(columns='Position')
# %% Define X and y variables
y = df['Salary']
X = df.drop(columns='Salary')
print(X.head())
# %% Calculating the exponentials of the Level Column
X['Level1'] = X['Level']**2
X['Level2'] = X['Level']**3
print(X.head())
# %% Normalize the Data
# Divide each column by it's maximum value
m = len(X)
X = X/X.max()
# %% Hypothesis Function
def hypothesis(X, theta):
    y1 = theta * X
    return np.sum(y1, axis = 1)
# %% Cost Function
def cost(X, y, theta):
    y1 = hypothesis(X, theta)
    return sum(np.sqrt((y1 - y) ** 2))/(2 * m)
# %% Gradient Descent Function
def gradientDescent(X, y, theta, alpha, epoch):
    J = []
    k = 0
    while k < epoch:
        y1 = hypothesis(X, theta)
        for c in range(0, len(X.columns)):
            theta[c] = theta[c] - alpha*sum((y1 - y) * X.iloc[:, c])/m
        j = cost(X, y, theta)
        J.append(j)
        k += 1
    return J, theta
# %% Initializing Parameters
theta = np.array([0.0]*len(X.columns))
# Calculating J and Theta
J, theta = gradientDescent(X, y, theta, 0.05, 700)
# %% Salary prediction
y_hat = hypothesis(X, theta)
# %% Original vs Predicted Salary Plot
%matplotlib inline
import matplotlib.pyplot as plt
plt.figure()
plt.scatter(x=X['Level'],y= y)           
plt.scatter(x=X['Level'], y=y_hat)
plt.show()
# %% Cost - At each Epoch
plt.figure()
plt.scatter(x=list(range(0, 700)), y = J)
plt.show()