from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from cvxopt import matrix
from cvxopt import solvers
from numpy import linalg
from scipy.spatial.distance import pdist

data_test = pd.read_csv('/home/shreya/Sem6/COL774/Assgn2.2/fashion_mnist/test.csv', header = None) 
data_train = pd.read_csv('/home/shreya/Sem6/COL774/Assgn2.2/fashion_mnist/train.csv', header = None) 


data_test = data_test.to_numpy()
data_train = data_train.to_numpy()


train_size = len(data_train)
test_size = len(data_test)
f_size = len(data_train[0]) - 1

# train_index = -np.ones(10,train_size)
# test_index = -np.ones(10,test_size)
X1 = []
X2 = []
X3 = []
X4 = []
X5 = []
X6 = []
X7 = []
X8 = []
X9 = []
X10 = []

X_train = []
Y_train = []
X_test = []
Y_test = []

for data in data_train:
    if data[f_size]==1:
        X1.append(data[0:f_size])
    elif data[f_size]==2:
        X2.append(data[0:f_size])
    elif data[f_size]==3:
        X3.append(data[0:f_size])
    elif data[f_size]==4:
        X4.append(data[0:f_size])
    elif data[f_size]==5:
        X5.append(data[0:f_size])
    elif data[f_size]==6:
        X6.append(data[0:f_size])
    elif data[f_size]==7:
        X7.append(data[0:f_size])
    elif data[f_size]==8:
        X8.append(data[0:f_size])
    elif data[f_size]==9:
        X9.append(data[0:f_size])
    elif data[f_size]==0:
        X10.append(data[0:f_size])

X_train.append(np.asarray(X1)/255)
X_train.append(np.asarray(X2)/255)
X_train.append(np.asarray(X3)/255)
X_train.append(np.asarray(X4)/255)
X_train.append(np.asarray(X5)/255)
X_train.append(np.asarray(X6)/255)
X_train.append(np.asarray(X7)/255)
X_train.append(np.asarray(X8)/255)
X_train.append(np.asarray(X9)/255)
X_train.append(np.asarray(X10)/255)

        
for data in data_test:
    X_test.append(data[0:f_size])
    Y_test.append(data[f_size])
        

X_test = np.asarray(X_test)/255
Y_test = np.asarray(Y_test)


# In[31]:


print(X_test[9].shape)


# In[32]:


# Y_train = np.reshape(Y_train, (train_size,1))
Y_test = np.reshape(Y_test, (test_size,1))
print(Y_test.shape)


# In[33]:


m = 4500


# In[8]:


#define q:
q = np.ones((m,1)) 
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
# print(h)

#define b:
b = matrix([0], tc='d')



# In[9]:


sigma = 10
def K_func(x, z):
    norm = np.linalg.norm(x-z)
    ans = exp(-0.5*(norm*norm)/sigma)
    return ans


# In[10]:


Y_temp = pdist([[1],[2],[3],[4]], 'euclidean')
print(Y_temp)


# In[34]:


print(type(Y_temp))
print(Y_temp[1])
print(Y_temp.shape)


# In[12]:


#define P for Gaussian kernel
def get_P(X_train, Y_train):
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

    # P = np.matmul(M, M.T)
    # print(P.shape)
    P = matrix(M, tc='d')
    return P
    # print(P.shape)
    


# In[13]:


#define A:
def get_A(Y_train):
    A = Y_train.T
    A = matrix(A, tc='d')
    return A


# In[14]:


def hypothesis(x, alpha, X_train, Y_train, support_vect):  #w*x
    ans = 0
    for i in support_vect:
        a = alpha[i]
        y = Y_train[i]
        ans = ans + (a*y*K_func(X_train[i,:], x))
    return ans


# In[15]:


def get_b(alpha, X_train, Y_train, support_vect):
    max1 = -1000000
    min1 =  1000000
    val = 0
    for i in range(0,m):
        val = hypothesis(X_train[i,:], alpha, X_train, Y_train, support_vect)
        if Y_train[i]>0:
            if (min1 > val):
                min1 = val
        else:
            if (max1 < val):
                max1 = val
    b_intercept = (-0.5)*(max1+min1)
    return (b_intercept)


# In[16]:


def get_sv(alpha):
    t_hold = 0.001
    index = []
    for i in range(0, len(alpha)):
        if alpha[i] > t_hold:
            index.append(i)
    return index


# In[17]:


def classify(X_train, Y_train, X_test):
    P = get_P(X_train, Y_train)
    A = get_A(Y_train)
    
    sol = solvers.qp(P,q,G,h,A,b)
    alpha = array(sol['x'])
    support_vect = get_sv(alpha)
#     print("alpha : ", alpha)
    print("SV: ", len(support_vect))
    
    b_intercept = get_b(alpha, X_train, Y_train, support_vect)
    print("b is: ", b_intercept)
    predict_list = []
    for i in range (0, test_size):
        predict = (hypothesis(X_test[i,:], alpha, X_train, Y_train, support_vect) + b_intercept)
#         print("predict: ", i, " is ", predict)
        predict_list.append(predict)
#         if predict > 0:
#             predict_list.append(1)
#         else:
#             predict_list.append(-1)

    
    return predict_list
    


# In[18]:


def max_votes(vote_list):
    maxv = 0
    c = 0
    for i in range(0, len(vote_list)):
        if maxv < vote_list[i]:
            maxv = vote_list[i]
            c = i
    return i


# In[19]:


def sigmoid(x):
    return 1/(1+exp(-x))



votes_class = np.zeros((test_size,10))
Y_train = np.vstack((-np.ones((2250,1)), np.ones((2250,1))))

# svm_classify = classify(np.vstack((X_train[2],X_train[3])), Y_train, X_test)
svm_classify_table = []

for i in range(0,9):
    for j in range(i+1,10):
        print("For index i ", i," j " , j)
        svm_classify = classify(np.vstack((X_train[i],X_train[j])), Y_train, X_test)
        svm_classify_table.append(svm_classify)
        for k in range(0,test_size):
            if svm_classify[k]>0:
                votes_class[k][j] += sigmoid(abs(svm_classify[k]))
                votes_class[k][i] += (1 - sigmoid(abs(svm_classify[k])))
            else:
                votes_class[k][i] += sigmoid(abs(svm_classify[k]))
                votes_class[k][j] += (1 - sigmoid(abs(svm_classify[k])))


# In[39]:


print(votes_class)


# In[52]:


def max_index(l):
    ind = 0
    max = l[0]
    for i in range(1, len(l)):
        if max < l[i]:
            max = l[i]
            ind = i
    return ind

accuracy = 0.0
Y_predict = []
predict = 0
for i in range (0, test_size):
    predict = max_index(votes_class[i]) + 1
#     print(predict)
    if predict==10: 
        predict = 0
#     print(f" Predict {predict} and Y = {Y_test[i]} \n")
    Y_predict.append(predict)
    if (predict == Y_test[i]):
        accuracy += 1.0

print(accuracy/test_size)
print(Y_predict)


int(Y_test[0])

#confusion matrix
confusion = np.zeros((10,10))

for i in range (0, len(Y_predict)):
    confusion[int(Y_test[i])][int(Y_predict[i])] += 1
print(confusion)


calculate test accuracy:
accuracy = 0.0
predict = 0
for i in range (0, test_size):
    predict = (hypothesis(X_test[i,:]) + b_intercept)
    if (predict * Y_test[i]) > 0:
        accuracy += 1.0

print("Accuracy: ", accuracy/test_size)

