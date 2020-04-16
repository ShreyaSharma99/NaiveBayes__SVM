from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from cvxopt import matrix
from cvxopt import solvers
from time import time

data_test = pd.read_csv('/home/shreya/Sem6/COL774/Assgn2.2/fashion_mnist/val.csv', header = None) 
data_train = pd.read_csv('/home/shreya/Sem6/COL774/Assgn2.2/fashion_mnist/train.csv', header = None) 

data_test = data_test.to_numpy()
data_train = data_train.to_numpy()


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
# print(Y_test.shape)


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


# In[9]:

t0 = time()
#define P = [y*x]*[y*x]T
M = []
for i in range(0, m):
    l = (Y_train[i]*X_train[i, :])
    M.append(l)

M = array(M)
# print(M.shape)
P = np.matmul(M, M.T)
# print(P.shape)
P = matrix(P, tc='d')
# print(P.shape)

#define q:
q = np.ones(train_size) 
q = matrix(-q, tc='d')

#define G:
G = np.eye(m)
G = np.vstack((-G, G))
G = matrix(G, tc='d')
# print(G)

#define h:
h0 = np.zeros((m,1))
h1 = np.ones((m,1))
h = matrix(np.vstack((h0, h1)), tc='d')
# print(h)

#define A:
A = Y_train.T
A = matrix(A, tc='d')

#define b:
b = matrix([0], tc='d')


sol = solvers.qp(P,q,G,h,A,b)


#calculate w = Sigma(alpha(i)*y(i)*x(i)):
print(sol['primal objective'])
alpha = array(sol['x'])
w = np.zeros((len(X_train[0]),1))
# print(w.shape)
for i in range (0, m):
    a = alpha[i]
    y = Y_train[i]
    w = w + np.reshape((a*y*X_train[i,:]),(len(X_train[0]),1))

#calculate b:
max1 = -1000000
min1 =  1000000
val = 0
for i in range (0, m):
    val = np.matmul(w.T, X_train[i,:])
    if Y_train[i]>0:
        if (min1 > val):
            min1 = val
    else:
        if (max1 < val):
            max1 = val
b = (-0.5)*(max1+min1)
print("b is: \n", b)

print(f'Training time (test): {round(time()-t0, 3)}s')


threshold = 0.00001
num_sv = 0

for i in range(0, m):
    if alpha[i]>threshold:
        num_sv += 1

print("num SV is: \n", num_sv)


# In[23]:


#calculate test accuracy:
accuracy = 0.0
predict = 0
for i in range (0, test_size):
    predict = (np.matmul(X_test[i,:], w) + b)
    if (predict * Y_test[i]) > 0:
        accuracy += 1.0

print("Accuracy test: \n", accuracy/test_size)


# In[24]:


#calculate train accuracy:
accuracy = 0.0
predict = 0
for i in range (0, train_size):
    predict = (np.matmul(X_train[i,:], w) + b)
    if (predict * Y_train[i]) > 0:
        accuracy += 1.0

print("Accuracy train: \n", accuracy/train_size)
# print(accuracy/train_size)



# In[2]:


from sklearn.svm import SVC
from numpy import linalg 

t0 = time()
clf = SVC(kernel='linear', C=1.0)
clf.fit(X_train, ravel(Y_train))
print(f'Training time (test): {round(time()-t0, 3)}s')

accuracy_svm = 0.0
# predict = 0
predict = clf.predict(X_test)
for i in range (0, test_size):
#     predict = clf.predict(np.reshape(X_test[i,:], (f_size,1)))
    if (predict[i] * Y_test[i]) >= 0:
        accuracy_svm += 1.0
print(accuracy_svm/test_size)

w_svm = clf.coef_
# print('w = ',w)
b_svm = clf.intercept_
print('b = ',b)
num_svs = clf.support_.shape



# In[5]:


print((w_svm).shape[1])
print((w).shape)


# In[6]:


w_svm = np.reshape(w_svm,(w_svm.shape[1],1))
print(w_svm.shape)
print('Indices of support vectors = ', clf.support_.shape)
print('Support vectors = ', clf.support_vectors_.shape)

diff = 0.0
for i in range(0, len(w)):
    diff += (w[i]-w_svm[i])*(w[i]-w_svm[i])
diff = np.sqrt(diff)
print("Diferrence in w = ", diff)
print("Diferrence in b = ", abs(b - b_svm))

