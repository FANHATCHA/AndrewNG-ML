import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat# Use loadmat to load matlab files
mat=loadmat(r"AndrewNG ML\machine-learning-ex3\ex3\ex3data1.mat")
#loadmat load .mat files as a dictionary with variable names as keys
X=mat["X"]
y=mat["y"]
X.shape#(5000,400) means 5000 training samples with 400 features becoz 20X20 pixel

import matplotlib.image as mpimg
_, axis = plt.subplots(10,10,figsize=(10,10))
for i in range(10):
    for j in range(10):
        axis[i,j].imshow(X[np.random.randint(0,5001),:].reshape(20,20,order="F") ) #reshape back to 20 pixel by 20 pixel
        axis[i,j].axis("off") #order=f to ensure the image is upright, axis off to remove axis coordinates

def sigmoid(z):
    return 1/(1+np.exp(-z))

def costFunctionReg(theta, X, y ,Lambda):
    m=len(y)
    predictions = sigmoid(X @ theta)
    cost = (1/m)*np.sum((-y * np.log(predictions)) - ((1-y)*np.log(1-predictions)))
    regCost= cost + Lambda/(2*m) * np.sum(theta[1:]**2)
    #compute gradient
    j_0= 1/m * (X.transpose() @ (predictions - y))[0]
    j_1 = 1/m * (X.transpose() @ (predictions - y))[1:] + (Lambda/m)* theta[1:]
    grad= np.vstack((j_0[:,np.newaxis],j_1))
    return regCost, grad

def gradientDescent(X,y,theta,alpha,num_iters,Lambda):
    m=len(y)
    J_history =[]
    for i in range(num_iters):
        cost, grad = costFunctionReg(theta,X,y,Lambda)
        theta = theta - (alpha * grad)
        J_history.append(cost)
    return theta , J_history

def oneVsAll(X, y, num_labels, Lambda): #iterates through to all the classes
    m,n=X.shape
    initial_theta = np.zeros((n+1,1))
    all_theta = [] #stores array initial_theta for evey classifier
    all_J=[] #similarly for J, i.e, the cost
    # add intercept terms
    X = np.hstack((np.ones((m,1)),X))

    for i in range(1,num_labels+1): #numlabel=10: digits 0 to 9
        theta , J_history = gradientDescent(X,np.where(y==i,1,0),initial_theta,1,300,Lambda)
        #np.where:vector of y with 1/0 for each class to conduct classification task within each iteration.
        all_theta.extend(theta)
        all_J.extend(J_history)
    return np.array(all_theta).reshape(num_labels,n+1), all_J

def predictOneVsAll(all_theta, X):
    m= X.shape[0]
    X = np.hstack((np.ones((m,1)),X))
    predictions = X @ all_theta.T
    return np.argmax(predictions,axis=1)+1

all_theta, all_J=oneVsAll(X,y,10,1)
pred = predictOneVsAll(all_theta, X)
print("Accuracy:",sum(pred[:,np.newaxis]==y)[0]/5000 *100,"%")

plt.plot(all_J[0:300])
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")
#-------------------------------------------------------------------------------

mat2=loadmat(r"AndrewNG ML\machine-learning-ex3\ex3\ex3weights.mat")
Theta1=mat2["Theta1"] # Theta1 has size 25 x 401
Theta2=mat2["Theta2"] # Theta2 has size 10 x 26

#feedforward propagation, i.e, already computed theta given
def predict(Theta1, Theta2, X):
    #Predict the label of an input given a trained neural network
    m= X.shape[0]
    X = np.hstack((np.ones((m,1)),X))
    a1 = sigmoid(X @ Theta1.T)
    a1 = np.hstack((np.ones((m,1)), a1)) # hidden layer
    a2 = sigmoid(a1 @ Theta2.T) # output layer
    return np.argmax(a2,axis=1)+1

pred2 = predict(Theta1, Theta2, X)
print("Training Set Accuracy:",sum(pred2[:,np.newaxis]==y)[0]/5000*100,"%")

#-------------------------------------------------------------------------------
