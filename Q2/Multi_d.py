from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import train_test_split


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


X_train = X_train/255
X_test = X_test/255


accuracy_train = np.zeros(5)
accuracy = np.zeros(5)


# In[8]:
from sklearn.model_selection import KFold # import KFold

kf = KFold(n_splits=5) # Define the split - into 2 folds 
kf.get_n_splits(X_train) # returns the number of splitting iterations in the cross-validator

C_list = [0.00001, 0.001, 1, 5, 10]

def get_accuracy(Y_predict, Y_test):
    accuracy = 0.0
    for i in range (0, len(Y_test)):
        if (Y_predict[i] == Y_test[i]):
            accuracy += 1.0
    print(accuracy/len(Y_test))

k_score = np.zeros((5,5))
count_f = 0
for train_index, test_index in kf.split(X_train):
    print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test = X_train[train_index], X_train[test_index]
    y_train, y_test = Y_train[train_index], Y_train[test_index]
    for i in range(0,5):
        clf = SVC(decision_function_shape='ovo',C=C_list[i], gamma=0.05)
        clf.fit(x_train, y_train)
        predict = clf.predict(x_test)
        k_score[i][count_f] = (get_accuracy(predict, y_test))
    count_f += 1

print(k_score)

accuracy_test = np.zeros(5)
for i in range(0,5):
        clf = SVC(decision_function_shape='ovo',C=C_list[i], gamma=0.05)
        clf.fit(X_train, Y_train)
        predict = clf.predict(X_test)
        accuracy_test[i] = (get_accuracy(predict, Y_test))

print(accuracy_test)

exit()
