import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
mat=loadmat(r"AndrewNG ML\machine-learning-ex6\ex6\ex6data1.mat")
X=mat["X"]
y=mat["y"]
m,n = X.shape

#better to use library rather than hard coding
from sklearn.svm import SVC
classifier = SVC(kernel="linear") #by default, C=1
classifier.fit(X,np.ravel(y)) #ravel flattens y as mx1 array

#plotting dataset
mask=y==1
plt.figure(figsize=(8,6)) #for output graph size
plt.scatter(X[mask[:,0],0],X[mask[:,0],1], label='positive', marker='+', s=50)
plt.scatter(X[~mask[:,0],0],X[~mask[:,0],1], label='negative', marker='o', s=50)
plt.legend()
# plotting the decision boundary with default C=1.0
X_1,X_2 = np.meshgrid(np.linspace(X[:,0].min(),X[:,0].max(),num=1000),np.linspace(X[:,1].min(),X[:,1].max(),num=1000), sparse=False)
plt.contour(X_1,X_2,classifier.predict(np.array([X_1.ravel(),X_2.ravel()]).T).reshape(X_1.shape[0],X_2.shape[0]), 1, colors="k") #1 is line width

#-------------------------------------------------------------------------------

# Test C = 100
classifier2 = SVC(C=100,kernel="linear")
classifier2.fit(X,np.ravel(y))
#plotting dataset
mask=y==1
plt.figure(figsize=(8,6)) #for output graph size
plt.scatter(X[mask[:,0],0],X[mask[:,0],1], label='positive', marker='+', s=50)
plt.scatter(X[~mask[:,0],0],X[~mask[:,0],1], label='negative', marker='o', s=50)
plt.legend()
# plotting the decision boundary
X_3,X_4 = np.meshgrid(np.linspace(X[:,0].min(),X[:,0].max(),num=1000),np.linspace(X[:,1].min(),X[:,1].max(),num=1000))
plt.contour(X_3,X_4,classifier2.predict(np.array([X_3.ravel(),X_4.ravel()]).T).reshape(X_3.shape),1,colors="k")

#-------------------------------------------------------------------------------
#Gaussian/rbf kernel
mat2 = loadmat(r"AndrewNG ML\machine-learning-ex6\ex6\ex6data2.mat")
X2 = mat2["X"]
y2 = mat2["y"]
m2,n2 = X2.shape

classifier3 = SVC(kernel="rbf",gamma=30,C=10) #gamma in rbf just like sigma in linear kernel
classifier3.fit(X2,y2.ravel())
#plotting dataset
mask2=y2==1
plt.figure(figsize=(8,6))
plt.scatter(X2[mask2[:,0],0],X2[mask2[:,0],1],c="r",marker="+")
plt.scatter(X2[~mask2[:,0],0],X2[~mask2[:,0],1],c="y",marker="o")
# plotting the decision boundary
X_5,X_6 = np.meshgrid(np.linspace(X2[:,0].min(),X2[:,0].max(),num=1000),np.linspace(X2[:,1].min(),X2[:,1].max(),num=1000))
plt.contour(X_5,X_6,classifier3.predict(np.array([X_5.ravel(),X_6.ravel()]).T).reshape(X_5.shape),1,colors="k")

#-------------------------------------------------------------------------------
#determining best C and gamma/sigma values
mat3 = loadmat(r"AndrewNG ML\machine-learning-ex6\ex6\ex6data3.mat")
X3 = mat3["X"]
y3 = mat3["y"]
Xval = mat3["Xval"]
yval = mat3["yval"]
m3,n3=X3.shape

def dataset3Params(X, y, Xval, yval,vals):
    acc = 0
    best_c=0
    best_gamma=0
    for i in vals:
        C= i
        for j in vals:
            gamma = 1/j
            classifier = SVC(C=C,gamma=gamma)
            classifier.fit(X,y)
            prediction = classifier.predict(Xval)
            score = classifier.score(Xval,yval)
            if score>acc:
                acc =score
                best_c =C
                best_gamma=gamma
    return best_c, best_gamma

vals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
C, gamma = dataset3Params(X3, y3.ravel(), Xval, yval.ravel(),vals)
classifier4 = SVC(C=C,gamma=gamma)
classifier4.fit(X3,y3.ravel())

#plotting dataset
mask3=y3==1
plt.figure(figsize=(8,6))
plt.scatter(X3[mask3[:,0],0],X3[mask3[:,0],1],c="r",marker="+")
plt.scatter(X3[~mask3[:,0],0],X3[~mask3[:,0],1],c="y",marker="o")
# plotting the decision boundary
X_7,X_8 = np.meshgrid(np.linspace(X3[:,0].min(),X3[:,0].max(),num=1000),np.linspace(X3[:,1].min(),X3[:,1].max(),num=1000))
plt.contour(X_7,X_8,classifier4.predict(np.array([X_7.ravel(),X_8.ravel()]).T).reshape(X_7.shape),1,colors="k")
