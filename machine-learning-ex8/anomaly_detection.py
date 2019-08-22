import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
mat=loadmat(r"AndrewNG ML\machine-learning-ex8\ex8\ex8data1.mat")
X=mat["X"]
Xval=mat["Xval"]
yval=mat["yval"]
X.shape
plt.scatter(X[:,0],X[:,1],marker="x")
plt.xlim(0,30)
plt.ylim(0,30)
plt.xlabel("Latency (ms)")
plt.ylabel("Throughput (mb/s)")

def estimateGaussian(X):
    m = X.shape[0]
    sum_ = np.sum(X,axis=0)
    mu = 1/m *sum_ #mean
    var = 1/m * np.sum((X - mu)**2,axis=0) #variance
    return mu,var

def multivariateGaussian(X, mu, sigma2): #computes probability density function
    k = len(mu)
    sigma2=np.diag(sigma2)
    X = X-mu.T #x=(x-mu).T should be
    p = 1/((2*np.pi)**(k/2)*(np.linalg.det(sigma2)**0.5))* np.exp(-0.5* np.sum(X @ np.linalg.pinv(sigma2) * X,axis=1))
    return p

mu, sigma2 = estimateGaussian(X) #sigma2=sigma squared
"""
#alternate way to calculate mean and sigma2
mu = np.mean(X, axis=0) # mean
sigma2 = np.cov(X, rowvar=0) # covariance matrix
from scipy.stats import multivariate_normal
p=multivariate_normal(X,mean=mu,cov=sigma2)
"""
p = multivariateGaussian(X, mu, sigma2)
# NOTE: contour plots not working
plt.figure(figsize=(8,6))
plt.scatter(X[:,0],X[:,1],marker="x")
X1,X2 = np.meshgrid(np.linspace(0,30,num=100),np.linspace(0,30,num=100))
p2 = multivariateGaussian(np.hstack((X1.flatten()[:,np.newaxis],X2.flatten()[:,np.newaxis])), mu, sigma2)
contour_level = 10**np.array([np.arange(-20,0,3,dtype=np.float)]).T
plt.contour(X1,X2,p2[:,np.newaxis].reshape(X1.shape),contour_level)
plt.xlim(0,30)
plt.ylim(0,30)
plt.xlabel("Latency (ms)")
plt.ylabel("Throughput (mb/s)")

def selectThreshold(yval, pval): #computes epsilon for selecting outliers
    best_epi = 0
    best_F1 = 0
    stepsize = (max(pval) -min(pval))/1000
    epi_range = np.arange(pval.min(),pval.max(),stepsize)
    for epi in epi_range:
        predictions = (pval<epi)[:,np.newaxis]
        tp = np.sum(predictions[yval==1]==1)
        fp = np.sum(predictions[yval==0]==1)
        fn = np.sum(predictions[yval==1]==0)
        # compute precision, recall and F1
        prec = tp/(tp+fp)
        rec = tp/(tp+fn)
        F1 = (2*prec*rec)/(prec+rec)
        if F1 > best_F1:
            best_F1 =F1
            best_epi = epi
    return best_epi, best_F1

pval = multivariateGaussian(Xval, mu, sigma2)
epsilon, F1 = selectThreshold(yval, pval)
print("Best epsilon found using cross-validation:",epsilon)
print("Best F1 on Cross Validation Set:",F1)

plt.figure(figsize=(8,6))
plt.scatter(X[:,0],X[:,1],marker="x")
X1,X2 = np.meshgrid(np.linspace(0,35,num=70),np.linspace(0,35,num=70))
p2 = multivariateGaussian(np.hstack((X1.flatten()[:,np.newaxis],X2.flatten()[:,np.newaxis])), mu, sigma2)
contour_level = 10**np.array([np.arange(-20,0,3,dtype=np.float)]).T
plt.contour(X1,X2,p2[:,np.newaxis].reshape(X1.shape),contour_level)
outliers = np.nonzero(p<epsilon)[0]#Circling of anomalies
plt.scatter(X[outliers,0],X[outliers,1],marker ="o",facecolor="none",edgecolor="r",s=70)
plt.xlim(0,35)
plt.ylim(0,35)
plt.xlabel("Latency (ms)")
plt.ylabel("Throughput (mb/s)")

#for high dimension dataset
mat2 = loadmat(r"AndrewNG ML\machine-learning-ex8\ex8\ex8data2.mat")
X2 = mat2["X"]
Xval2 = mat2["Xval"]
yval2 = mat2["yval"]
mu2, sigma2_2 = estimateGaussian(X2)
p3 = multivariateGaussian(X2, mu2, sigma2_2)
pval2 = multivariateGaussian(Xval2, mu2, sigma2_2)
epsilon2, F1_2 = selectThreshold(yval2, pval2)
print("Best epsilon found using cross-validation:",epsilon2)
print("Best F1 on Cross Validation Set:",F1_2)
print("# Outliers found:",np.sum(p3<epsilon2))
