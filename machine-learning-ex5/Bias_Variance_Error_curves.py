import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
mat=loadmat(r"AndrewNG ML\machine-learning-ex5\ex5\ex5data1.mat")
X=mat["X"]
y=mat["y"]
Xtest=mat["Xtest"]
ytest=mat["ytest"]
Xval=mat["Xval"]
yval=mat["yval"]

plt.scatter(X,y,marker="x",color="r")
plt.scatter(Xtest,ytest)
plt.scatter(Xval,yval)
plt.xlabel("Change in water level")
plt.ylim(0,40)
plt.ylabel("Water flowing out of the dam")

def linearRegCostFunction(X, y,theta, Lambda):
    m = len(y)
    predictions = X @ theta
    cost = 1/(2*m) * np.sum((predictions - y)**2)
    reg_cost = cost + Lambda/(2*m) * (np.sum(theta[1:]**2))
    # compute the gradient
    grad1 = 1/m * X.T @ (predictions - y)
    grad2 = 1/m * X.T @ (predictions - y) + (Lambda/m * theta)
    grad = np.vstack((grad1[0],grad2[1:]))
    return reg_cost, grad

def gradientDescent(X,y,theta,alpha,num_iters,Lambda):
    m=len(y)
    J_history =[]
    for i in range(num_iters):
        cost, grad = linearRegCostFunction(X,y,theta,Lambda)
        theta = theta - (alpha * grad)
        J_history.append(cost)
    return theta, J_history

m,n = X.shape
Lambda=0
theta = np.zeros((n+1,1))
X_1 = np.hstack((np.ones((m,1)),X))
theta, J_history = gradientDescent(X_1,y,theta,0.001,4000,Lambda)
plt.plot(J_history)

plt.scatter(X,y,marker="x",color="r")
plt.xlabel("Change in water level")
plt.ylabel("Water flowing out of the dam")
x_value=[x for x in range(-50,40)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="b")
plt.ylim(-5,40)
plt.xlim(-50,40)

def learningCurve(X, y, Xval, yval, Lambda):#gives training and validation set error
    m=len(y)
    err_train, err_val = [],[]
    for i in range(1,m+1):
        theta = gradientDescent(X[0:i,:],y[0:i,:],np.zeros((n+1,1)),0.001,3000,Lambda)[0]
        err_train.append(linearRegCostFunction(X[0:i,:], y[0:i,:], theta, Lambda)[0])
        err_val.append(linearRegCostFunction(Xval, yval, theta, Lambda)[0])
    return err_train, err_val

Xval_1 = np.hstack((np.ones((21,1)),Xval))
error_train, error_val = learningCurve(X_1, y, Xval_1, yval, Lambda)

plt.plot(range(12),error_train,label="Train")
plt.plot(range(12),error_val,label="Cross Validation",color="r")
plt.title("Learning Curve for Linear Regression")
plt.xlabel("Number of training examples")
plt.ylabel("Error")
plt.legend()

#-------------------------------------------------------------------------------
#polynomial Regression

def polyFeatures(X, p):# basically to increase features
    #X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
    for i in range(2,p+1):
        X = np.hstack((X,(X[:,0]**i)[:,np.newaxis]))
    return X

# Map X onto Polynomial features and normalize
p=8
X_poly = polyFeatures(X, p)
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_poly=sc_X.fit_transform(X_poly)
X_poly = np.hstack((np.ones((X_poly.shape[0],1)),X_poly))
# Map Xtest onto polynomial features and normalize
X_poly_test = polyFeatures(Xtest, p)
X_poly_test = sc_X.transform(X_poly_test)
X_poly_test = np.hstack((np.ones((X_poly_test.shape[0],1)),X_poly_test))
# Map Xval onto polynomial features and normalize
X_poly_val = polyFeatures(Xval, p)
X_poly_val = sc_X.transform(X_poly_val)
X_poly_val = np.hstack((np.ones((X_poly_val.shape[0],1)),X_poly_val))

theta_poly, J_history_poly = gradientDescent(X_poly,y,np.zeros((9,1)),0.3,500,Lambda)
plt.scatter(X,y,marker="x",color="r")
plt.xlabel("Change in water level")
plt.ylabel("Water flowing out of the dam")
x_value=np.linspace(-55,65,200)
# Map the X values and normalize
x_value_poly = polyFeatures(x_value[:,np.newaxis], p)
x_value_poly = sc_X.transform(x_value_poly)
x_value_poly = np.hstack((np.ones((x_value_poly.shape[0],1)),x_value_poly))
y_value= x_value_poly @ theta_poly
plt.plot(x_value,y_value,"--",color="b")

# error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval, Lambda)
# plt.plot(range(12),error_train,label="Train")
# plt.plot(range(12),error_val,label="Cross Validation",color="r")
# plt.title("Learning Curve for Linear Regression")
# plt.xlabel("Number of training examples")
# plt.ylabel("Error")
# plt.legend()
