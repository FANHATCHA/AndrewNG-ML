import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from nltk.stem import PorterStemmer
file_contents = open(r"AndrewNG ML\machine-learning-ex6\ex6\emailSample1.txt","r").read()
vocabList = open(r"AndrewNG ML\machine-learning-ex6\ex6\\vocab.txt","r").read()
vocabList=vocabList.split("\n")[:-1]
vocabList_d={}
for ea in vocabList:
    value,key = ea.split("\t")[:]
    vocabList_d[key] = value

def processEmail(email_contents,vocabList_d):#preprocesses and returns indicies of words
    # Lower case
    email_contents = email_contents.lower()
    # Handle numbers
    email_contents = re.sub("[0-9]+","number",email_contents)
    # Handle URLS
    email_contents = re.sub("[http|https]://[^\s]*","httpaddr",email_contents)
    # Handle Email Addresses
    email_contents = re.sub("[^\s]+@[^\s]+","emailaddr",email_contents)
    # Handle $ sign
    email_contents = re.sub("[$]+","dollar",email_contents)
    # Strip all special characters
    specialChar = ["<","[","^",">","+","?","!","'",".",",",":"]
    for char in specialChar:
        email_contents = email_contents.replace(str(char),"")
    email_contents = email_contents.replace("\n"," ")
    # Stem the word
    ps = PorterStemmer()
    email_contents = [ps.stem(token) for token in email_contents.split(" ")]
    email_contents= " ".join(email_contents)
    # Process the email and return word_indices
    word_indices=[]
    for char in email_contents.split():
        if len(char)>1 and char in vocabList_d:
            word_indices.append(int(vocabList_d[char]))
    return word_indices

word_indices= processEmail(file_contents,vocabList_d)

def emailFeatures(word_indices, vocabList_d):#converting to feature vector
    n = len(vocabList_d)
    features = np.zeros((n,1))
    for i in word_indices:
        features[i] =1
    return features

features = emailFeatures(word_indices,vocabList_d)
print("Length of feature vector: ",len(features))
print("Number of non-zero entries: ",np.sum(features))

from scipy.io import loadmat
spam_mat = loadmat(r"AndrewNG ML\machine-learning-ex6\ex6\spamTrain.mat")
X_train =spam_mat["X"]
y_train = spam_mat["y"]

from sklearn.svm import SVC
spam_svc = SVC(C=0.1,kernel ="linear")
spam_svc.fit(X_train,y_train.ravel())
print("Training Accuracy:",(spam_svc.score(X_train,y_train.ravel()))*100,"%")

spam_mat_test = loadmat(r"AndrewNG ML\machine-learning-ex6\ex6\spamTest.mat")
X_test = spam_mat_test["Xtest"]
y_test =spam_mat_test["ytest"]
spam_svc.predict(X_test)
print("Test Accuracy:",(spam_svc.score(X_test,y_test.ravel()))*100,"%")

#weight of a words most predicitive of spam mail
weights = spam_svc.coef_[0]
weights_col = np.hstack((np.arange(1,1900).reshape(1899,1),weights.reshape(1899,1)))
df = pd.DataFrame(weights_col)
df.sort_values(by=[1],ascending = False,inplace=True)
predictors = []
idx=[]
for i in df[0][:15]:
    for keys, values in vocabList_d.items():
        if str(int(i)) == values:
            predictors.append(keys)
            idx.append(int(values))
print("Top predictors of spam:")
for _ in range(15):
    print(predictors[_],"\t\t",round(df[1][idx[_]-1],6))
