import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv(r'AndrewNG ML/machine-learning-ex1/data.txt', header=None)
data.head()
data.describe() # gives stats of dataset
X = data.iloc[:,0] #all features
y = data.iloc[:,1]
m=len(y) #training examples , i.e, total training set
X = (X - np.mean(X))/np.std(X) #feature scaling, data2.txt
plt.scatter(X,y)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.show()

X = X[:,np.newaxis] #newaxis adds another column making matrix multiplication easier
y = y[:,np.newaxis]
theta=np.zeros([2,1]) #create 2X1 array of 0s. Has both values theta0 and theta1
iterations = 2000 # see LOC 29 trying to reach minimum J()
alpha = 0.01 #learning rate
ones = np.ones((m,1))
X = np.hstack((ones, X)) # adding the intercept term

def computeCost(X, y, theta): #squared mean error function
    hox = np.dot(X, theta)
    return np.sum(np.power((hox-y), 2)) / (2*m)

def gradientDescent(X, y, theta, alpha, iterations): #minimize J ,i.e, find best fit theta0 & theta1
    for _ in range(iterations):
        temp = np.dot(X, theta) - y
        temp = np.dot(X.T, temp) #temp gives derivative of J
        theta = theta - ((alpha/m) * temp)# alt method:theta=((X.T*X)**(-1))*X.T*y
    return theta

theta = gradientDescent(X, y, theta, alpha, iterations)
J = computeCost(X, y, theta)
print(J)
plt.scatter(X[:,1],y)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.plot(X[:,1], np.dot(X, theta), color='k') #2nd parameter is line hox=theta0+theta1*x
plt.show()
