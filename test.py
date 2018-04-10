'''from keras import backend as K
import os

def set_keras_backend(backend):

    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
       # reload(K)
        assert K.backend() == backend

set_keras_backend("tensorflow")  '''

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
data=pd.read_csv("/home/piku/PycharmProjects/NSProject/dataset/finally.csv")
print(data.describe())
print(data.RESULTANT.dtype)
data['RESULTANT']=data.RESULTANT.map({"'A'":0,"'C'":1})
#print(data.isnull().sum().nunique())    #gives number of unique values
print(data.RESULTANT)
y=np.array(data["RESULTANT"])
x=np.array(data.iloc[:,:-1])
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.33, random_state=70)
#for kernel in ('linear','poly','rbf'):
clf=SVC(kernel='rbf')
clf.fit(xtrain,ytrain)
print(clf.score(xtest,ytest))
print("abc")    #just to test
