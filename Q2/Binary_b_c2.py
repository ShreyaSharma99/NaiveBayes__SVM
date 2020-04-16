from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from cvxopt import matrix
from cvxopt import solvers
from numpy import linalg
from scipy.spatial.distance import pdist
from time import time 


# In[2]:


data_test = pd.read_csv('/home/shreya/Sem6/COL774/Assgn2.2/fashion_mnist/val.csv', header = None) 
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
    if (data[f_size] == 3) or (data[f_size] == 4):
        X_test.append(data[0:f_size])
        if data[f_size]==3:
            Y_test.append(-1)
        else:
            Y_test.append(1)
        
for data in data_train:
    if (data[f_size] == 3) or (data[f_size] == 4):
        X_train.append(data[0:f_size])
        if data[f_size]==3:
            Y_train.append(-1)
        else:
            Y_train.append(1)
        
train_size = len(Y_train)
test_size = len(Y_test)

X_train = np.asarray(X_train)
Y_train = np.asarray(Y_train)
X_test = np.asarray(X_test)
Y_test = np.asarray(Y_test)


# In[5]:


Y_train = np.reshape(Y_train, (train_size,1))
Y_test = np.reshape(Y_test, (test_size,1))
print(Y_test.shape)


# In[6]:


#Data scaling from [0, 256] to [0, 1]
def scale_data(data):
    data = data/255
    return data


# In[7]:


X_train = scale_data(X_train)
X_test = scale_data(X_test)


# In[8]:


m = train_size


t0 = time()
#define P for Gaussian kernel
M = np.ones((m,m))
norm_list = pdist(X_train, 'euclidean')
count = 0
for i in range(0,m-1):
    for j in range(i+1,m):
#             temp = (Y_train[i]*Y_train[j])*K_func(X_train[i,:], X_train[j,:])
        temp = (Y_train[i]*Y_train[j])*np.exp(-0.05*(norm_list[count]*norm_list[count]))
        count += 1
        M[i][j] = temp
        M[j][i] = temp


P = matrix(M, tc='d')

sigma = 10
def K_func(x, z):
    norm = np.linalg.norm(x-z)
    ans = exp(-0.5*(norm*norm)/sigma)
    return ans


#define q:
q = np.ones(train_size) 
q = matrix(-q, tc='d')
# print(q)

#define G:
G = np.eye(m)
G = np.vstack((-G, G))
G = matrix(G, tc='d')
# print(G)

#define h:
h0 = np.zeros((m,1))
h1 = np.ones((m,1))
h = matrix(np.vstack((h0, h1)), tc='d')

#define A:
A = Y_train.T
A = matrix(A, tc='d')

#define b:
b = matrix([0], tc='d')


sol = solvers.qp(P,q,G,h,A,b)
print(f'Training time (test): {round(time()-t0, 3)}s')


#calculate w = Sigma(alpha(i)*y(i)*x(i)):
print(sol['primal objective'])
alpha = array(sol['x'])
w = np.zeros((len(X_train[0]),1))
print(w.shape)
for i in range (0, m):
    a = alpha[i]
    y = Y_train[i]
    w = w + np.reshape((a*y*X_train[i,:]),(len(X_train[0]),1))
print(w)
def hypothesis(x):
    ans = 0.0
    for i in range (0, m):
        a = alpha[i]
        if(a>0.0001):
            y = Y_train[i]
            ans = ans + (a*y*K_func(X_train[i,:], x))
    return ans


#calculate b:
max1 = -1000000
min1 =  1000000
val = 0
for i in range (0, m):
    val = hypothesis(X_train[i,:])
    if Y_train[i]>0:
        if (min1 > val):
            min1 = val
    else:
        if (max1 < val):
            max1 = val
b_intercept = (-0.5)*(max1+min1)
print(b_intercept)


# In[24]:


threshold = 0.01
num_sv = 0

for i in range(0, m):
    if alpha[i]>threshold:
        num_sv += 1
print(num_sv)

print(alpha)

#calculate test accuracy:
accuracy = 0.0
predict = 0
for i in range (0, test_size):
    predict = (hypothesis(X_test[i,:]) + b_intercept)
    if (predict * Y_test[i]) > 0:
        accuracy += 1.0

print(accuracy/test_size)


#####################################################################skLearn_Gaussian##############################################################################################

from sklearn.svm import SVC


# In[32]:


t0 = time()
model = SVC(C=1.0, gamma=0.05)
model.fit(X_train, ravel(Y_train))
print(f'Training time (test): {round(time()-t0, 3)}s')


# In[33]:


accuracy_svm = 0.0
# predict = 0
predict = model.predict(X_test)
for i in range (0, test_size):
#     predict = clf.predict(np.reshape(X_test[i,:], (f_size,1)))
    if (predict[i] * Y_test[i]) >= 0:
        accuracy_svm += 1.0
print(accuracy_svm/test_size)


print('Indices of support vectors = ', model.support_.shape)
print('Support vectors = ', model.support_vectors_.shape)


b_svm = model.intercept_
print("Diferrence in b = ", abs(b - b_svm))


# In[ ]:




