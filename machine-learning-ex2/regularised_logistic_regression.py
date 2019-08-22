import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv(r'AndrewNG ML\machine-learning-ex2\ex2data2.txt', header=None)
X = data.iloc[:,0:2]
y = data.iloc[:,2]

def mapFeature(X1, X2): #since decision boundary is non linear
    degree = 6
    out = np.ones(X.shape[0])[:,np.newaxis]#adds 1s col. also
    for i in range(1, degree+1):
        for j in range(i+1):
            out = np.hstack((out, (X1**(i-j) * X2**j)[:,np.newaxis]))
    return out

X = mapFeature(X.iloc[:,0], X.iloc[:,1])

def sigmoid(x):
    return 1/(1+np.exp(-x))

def costFunctionReg(theta, X, y ,Lambda):
    predictions = sigmoid(X @ theta)
    cost = (1/m)*np.sum((-y * np.log(predictions)) - ((1-y)*np.log(1-predictions)))
    regCost= cost + Lambda/(2*m) * np.sum(theta**2)
    #compute gradient
    j_0= 1/m * (X.transpose() @ (predictions - y))[0]
    j_1 = 1/m * (X.transpose() @ (predictions - y))[1:] + (Lambda/m)* theta[1:]
    grad= np.vstack((j_0[:,np.newaxis],j_1))
    return regCost, grad

(m, n) = X.shape
y=y[:,np.newaxis]
theta = np.zeros((n,1))
Lambda = 1

def gradientDescent(X,y,theta,alpha,num_iters,Lambda):
    m=len(y)
    J_history =[]
    for i in range(num_iters):
        cost, grad = costFunctionReg(theta,X,y,Lambda)
        theta = theta - (alpha * grad)
        J_history.append(cost)
    return theta , J_history

theta , J_history = gradientDescent(X,y,theta,1,800,0.2)
J=J_history[-1]

#plotting graph
def mapFeatureForPlotting(X1, X2):
    degree = 6
    out = np.ones(1)
    for i in range(1, degree+1):
        for j in range(i+1):
            out = np.hstack((out, np.multiply(np.power(X1, i-j), np.power(X2, j)))) #difference - no newaxis
    return out
u = np.linspace(-1, 1.5, 50)
v = np.linspace(-1, 1.5, 50)
z = np.zeros((len(u), len(v)))
for i in range(len(u)):
    for j in range(len(v)):
        z[i,j] = np.dot(mapFeatureForPlotting(u[i], v[j]), theta)
mask=y.flatten()==1
plt.scatter(X[mask][:,1], X[mask][:,2], label='Pass')
plt.scatter(X[~mask][:,1], X[~mask][:,2], label='Fail')
plt.contour(u,v,z,0, colors='k')
plt.xlabel('Microchip Test1')
plt.ylabel('Microchip Test2')
plt.legend()
plt.show()

#accuracy\
pred = [sigmoid(np.dot(X, theta)) >= 0.5]
print(np.mean(pred == y.flatten()) * 100)
