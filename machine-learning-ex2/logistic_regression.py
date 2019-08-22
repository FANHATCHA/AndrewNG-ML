import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv(r'AndrewNG ML\machine-learning-ex2\ex2data1.txt', header=None)
data.head()
X = data.iloc[:,0:2]
y = data.iloc[:,2]
m=len(y) #training examples , i.e, total training set
pos , neg = (y==1) , (y==0)
plt.scatter(X[pos][0],X[pos][1], label='Admitted', marker='+')
plt.scatter(X[neg][0],X[neg][1], label='Not_Admitted')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend()
plt.show()

def sigmoid(z):
  return 1/(1+np.exp(-z))

def costFunction(theta, X, y):
    #np.sum sums values inside the matrix
    J = (-1/m) * np.sum(y*np.log(sigmoid(X @ theta)) + (1-y)*np.log(1 - sigmoid(X @ theta))) #@ gives matrix multiplication
    return J

def gradient(theta, X, y):
    return ((1/m) * X.T @ (sigmoid(X @ theta) - y))

def featureNormalization(X):
    mean=np.mean(X)
    std=np.std(X)
    X_norm = (X - mean)/std
    return X_norm, mean, std

def gradientDescent(X,y,theta,alpha,num_iters):
    J_history=[]
    for _ in range(num_iters):
        cost = costFunction(theta, X, y)
        grad = gradient(theta,X,y)
        theta = theta - (alpha * grad)
        J_history.append(cost)
    return theta, J_history

X,X_mean,X_std=featureNormalization(X)
(m, n) = X.shape
X=np.hstack((np.ones((m,1)),X))
y = y[:,np.newaxis]
theta=np.zeros((n+1,1))
alpha=1
num_iters=500

theta, J=gradientDescent(X,y,theta,alpha,num_iters)
plt.plot([x for x in range(num_iters)],J) #J vs num_iters graph
J=J[-1] #we need last value

plt.scatter(X[pos][:,1],X[pos][:,2],c="r",marker="+",label="Admitted")
plt.scatter(X[neg][:,1],X[neg][:,2],c="b",marker="x",label="Not admitted")
x_value= np.array([np.min(X[:,1]),np.max(X[:,1])])
y_value=-(theta[0] +theta[1]*x_value)/theta[2]
plt.plot(x_value,y_value, "k", label='decision_boundary')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend()
plt.show()

#testing
x_test = np.array([45,85])
x_test = (x_test - X_mean)/X_std
x_test = np.append(np.ones(1),x_test)
prob = sigmoid(x_test.dot(theta))
print("We predict an admission probability of",prob[0])
#accuracy
def classifierPredict(theta,X):
    predictions = sigmoid(X.dot(theta))
    return predictions>=0.5
p=classifierPredict(theta,X)
print("Train Accuracy:", sum(p==y)[0])
