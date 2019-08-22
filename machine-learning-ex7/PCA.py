import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
mat3 = loadmat(r"AndrewNG ML\machine-learning-ex7\ex7\ex7data1.mat")
X3 = mat3["X"]
plt.scatter(X3[:,0],X3[:,1],marker="o",facecolors="yellow",edgecolors="k")

def featureNormalize(X):
    mu = np.mean(X,axis=0)
    sigma = np.std(X,axis=0)
    X_norm = (X - mu)/sigma
    return X_norm, mu , sigma

def pca(X):#computes eigen vectors of covariance matrix
    m,n = X.shape[0], X.shape[1]
    sigma = 1/m * X.T @ X
    U,S,V = svd(sigma)
    return U,S,V

from numpy.linalg import svd
X_norm,mu,std = featureNormalize(X3)
U,S = pca(X_norm)[:2]
#eigen vector Visualisation
plt.scatter(X3[:,0],X3[:,1],marker="o",facecolors="none",edgecolors="b")
plt.plot([mu[0],(mu+1.5*S[0]*U[:,0].T)[0]],[mu[1],(mu+1.5*S[0]*U[:,0].T)[1]],color="black",linewidth=3)
plt.plot([mu[0],(mu+1.5*S[1]*U[:,1].T)[0]],[mu[1],(mu+1.5*S[1]*U[:,1].T)[1]],color="black",linewidth=3)
print("Top eigenvector U(:,1) =:",U[:,0])

#data compression 2d to 1d,
def projectData(X, U, K):
    m = X.shape[0]
    U_reduced = U[:,:K]
    Z = np.zeros((m,K))
    for i in range(m):
        for j in range(K):
            Z[i,j] = X[i,:] @ U_reduced[:,j]
    return Z

# Project the data onto K=1 dimension
K=1
Z = projectData(X_norm, U, K)
print("Projection of the first example:",Z[0][0])

#data recovery approximation
def recoverData(Z, U, K):
    m,n = Z.shape[0],U.shape[0]
    X_rec = np.zeros((m,n))
    U_reduced = U[:,:K]
    for i in range(m):
        X_rec[i,:] = Z[i,:] @ U_reduced.T
    return X_rec

X_rec  = recoverData(Z, U, K)
print("Approximation of the first example:",X_rec[0,:])

#Visualisation
plt.scatter(X_norm[:,0],X_norm[:,1],marker="o",label="Original",facecolors="none",edgecolors="b",s=15)
plt.scatter(X_rec[:,0],X_rec[:,1],marker="o",label="Approximation",facecolors="none",edgecolors="r",s=15)
plt.title("The Normalized and Projected Data after PCA")
plt.legend()

#image compression using PCA
mat4 = loadmat(r"AndrewNG ML\machine-learning-ex7\ex7\ex7faces.mat")
X4 = mat4["X"]
fig, ax = plt.subplots(nrows=10,ncols=10,figsize=(8,8))
for i in range(0,100,10):
    for j in range(10):
        ax[int(i/10),j].imshow(X4[i+j,:].reshape(32,32,order="F"),cmap="gray")
        ax[int(i/10),j].axis("off")

X_norm2 = featureNormalize(X4)[0]
# Run PCA
U2 =pca(X_norm2)[0]
#Visualize the top 36 eigenvectors found that describe the largest variations in dataset
U_reduced = U2[:,:36].T
fig2, ax2 = plt.subplots(6,6,figsize=(8,8))
for i in range(0,36,6):
    for j in range(6):
        ax2[int(i/6),j].imshow(U_reduced[i+j,:].reshape(32,32,order="F"),cmap="gray")
        ax2[int(i/6),j].axis("off")

K2 = 100
Z2 = projectData(X_norm2, U2, K2)
print("The projected data Z has a size of:",Z2.shape)
# Data reconstruction
X_rec2  = recoverData(Z2, U2, K2)
# Visualize the reconstructed data
fig3, ax3 = plt.subplots(10,10,figsize=(8,8))
for i in range(0,100,10):
    for j in range(10):
        ax3[int(i/10),j].imshow(X_rec2[i+j,:].reshape(32,32,order="F"),cmap="gray")
        ax3[int(i/10),j].axis("off")
