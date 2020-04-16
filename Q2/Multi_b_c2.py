from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


# In[2]:


data_test = pd.read_csv('/home/shreya/Sem6/COL774/Assgn2.2/fashion_mnist/test.csv', header = None) 
data_train = pd.read_csv('/home/shreya/Sem6/COL774/Assgn2.2/fashion_mnist/train.csv', header = None) 


# In[3]:


data_test = data_test.to_numpy()
data_train = data_train.to_numpy()


# In[4]:


train_size = len(data_train)
test_size = len(data_test)
f_size = len(data_train[0]) - 1

X_test = []
Y_test = []
X_train = []
Y_train = []

for data in data_test:
    X_test.append(data[0:f_size])
    Y_test.append(data[f_size])
    
        
for data in data_train:
    X_train.append(data[0:f_size])
    Y_train.append(data[f_size])
    
        
train_size = len(Y_train)
test_size = len(Y_test)

X_train = np.asarray(X_train)
Y_train = np.asarray(Y_train)
X_test = np.asarray(X_test)
Y_test = np.asarray(Y_test)


# In[5]:


# Y_train = np.reshape(Y_train, (train_size,1))
# Y_test = np.reshape(Y_test, (test_size,1))
print(X_test.shape)
X_train = X_train/255
X_test = X_test/255


# In[6]:


clf = SVC(decision_function_shape='ovo',C=1.0, gamma=0.05)
clf.fit(X_train, Y_train)


# In[7]:


print(X_test.shape[1])


# In[8]:


accuracy = 0.0
# predict = 0
predict = clf.predict(X_test)
for i in range (0, test_size):
#     predict = clf.predict(np.reshape(X_test[i,:], (f_size,1)))
    if (predict[i] == Y_test[i]):
        accuracy += 1.0
print(accuracy/test_size)


# In[9]:


# print('w = ',clf.coef_)
print('b = ',clf.intercept_)
print('Indices of support vectors = ', clf.support_.shape)
print('Support vectors = ', clf.support_vectors_.shape)


# In[10]:


confusion_matrix(Y_test, predict)


# In[15]:


print(ravel(Y_test))

