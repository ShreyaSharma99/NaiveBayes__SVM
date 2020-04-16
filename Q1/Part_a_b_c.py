#!/usr/bin/env python
# coding: utf-8

# In[12]:


from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import re
import pandas as pd
# from nltk.tokenize import word_tokenize 


# In[13]:


#reading data function 
def read_data(file_name, flag):
#     f = io.open(file_name)
    f = pd.read_csv(file_name, header = None,  encoding='ISO-8859-1') 
    f = f.to_numpy() #this is a 1600000*6 numpy array now
    
    if flag:
        tweet_list0 = []
        tweet_list4 = []
        for tweet in f:
            string = re.split(r'[\s]+',tweet[5])
            if (tweet[0]==0):
#                 data.append(col[0])
                tweet_list0.append(string)
            elif (tweet[0]==4):
                tweet_list4.append(string)
        return tweet_list0, tweet_list4
    else:
        tweet_list = []
        data = []
        for tweet in f:
            string = re.split(r'[\s]+',tweet[5])
            if (tweet[0]!=2):
                tweet_list.append(string)
                data.append(tweet[0])
        return tweet_list, data


# In[14]:


#reading data from the training set of 1600000 samples
tweet_list0, tweet_list4 = read_data('/home/shreya/Sem6/COL774/Assgn2/trainingandtestdata/training.1600000.processed.noemoticon.csv', True)
print(len(tweet_list0))

#reading data from the testing set of 498 samples
test_list, data_test = read_data('/home/shreya/Sem6/COL774/Assgn2/trainingandtestdata/testdata.manual.2009.06.14.csv', False)
print(len(test_list))


# In[4]:


# import nltk
# from nltk.corpus import stopwords
# nltk.download('stopwords')
# stop_words = set(stopwords.words('english'))
# stop_words.add("\s")


# In[5]:


train_size0 = len(tweet_list0)
train_size4 = len(tweet_list4)

test_size = len(test_list)


# In[6]:


vocab = set()


# In[7]:


for tweet in tweet_list0:
    line = []
    for w in tweet:
#         if w not in stop_words: 
        line.append(w) 
        vocab.add(w)
    tweet = line
    
for tweet in tweet_list4:
    line = []
    for w in tweet:
#         if w not in stop_words: 
        line.append(w) 
        vocab.add(w)
    tweet = line


# In[8]:


print(len(vocab))


# In[9]:


Dict = {}
i = 0
for word in vocab:
    Dict[word] = i
    i +=1


# In[10]:


# 0 = negative, 2 = neutral, 4 = positive

p_y0 = float(train_size0)/(train_size0 + train_size4)
p_y4 = float(train_size4)/(train_size0 + train_size4)

print(p_y0)
print(p_y4)


# In[11]:


total_words_y0 = 0
total_words_y4 = 0

v = len(vocab)
p_x_y0 = np.ones(v)     #initialized by one for laplace smoothing
p_x_y4 = np.ones(v)

index = 0

for tweet in tweet_list0:
    for i in range(0, len(tweet)):
#         print(tweet[i])
        index = Dict.get(tweet[i])
        if index!=None:
            p_x_y0[index] +=1

for tweet in tweet_list4:
    for i in range(0, len(tweet)):
        index = Dict.get(tweet[i])
        if index!=None:
            p_x_y4[index] +=1

p_x_y0 = p_x_y0/(train_size0 + v)
p_x_y4 = p_x_y4/(train_size4 + v)


# In[12]:


print(Dict.get(tweet_list0[0][1]))
print(p_x_y0[2])


# In[13]:


def test_MLA(tweet):
    p0 = np.log(p_y0)
    p4 = np.log(p_y4)    
    index = 0
    v_len = v
    for word in tweet:
        index = Dict.get(word)
        if (index == None):
            v_len +=1
            p0 += np.log(1.0/(total_words_y0 + v))
            p4 += np.log(1.0/(total_words_y4 + v))
        else:
            p0 += np.log(p_x_y0[index])
            p4 += np.log(p_x_y4[index])
    
#     p0 = np.log(p0)
#     p4 = np.log(p4)
    
#     print("p0: ", p0)
#     print("p4: ", p4)
    
    if(p0>p4):
        return 0
    else:
        return 4


# In[23]:


def accuracy(test_list, data_test):
    confusion_matrix = np.zeros((2,2))  
    correct = 0.0
#     total = len(tweet_list)
    predict = ""
    for i in range(0, len(test_list)):
        predict = test_MLA(test_list[i])
        if(predict == 0):
            if(data_test[i]==predict):
                confusion_matrix[0][0] +=1
            else:
                confusion_matrix[0][1] +=1
        if(predict == 4):
            if(data_test[i]==predict):
                confusion_matrix[1][1] +=1
            else:
                confusion_matrix[1][0] +=1
        
    correct = confusion_matrix[0][0] + confusion_matrix[1][1]
    
    return (float(correct)/test_size), confusion_matrix   


# In[15]:


# def remove_stopwords(tweet_list):
#     new_list = []
#     for tweet in tweet_list:
#         line = []
#         for w in tweet:
#             if w not in stop_words:
#                 line.append(w)
#         new_list.append(line)
#     return new_list


# In[16]:


# test_list = remove_stopwords(test_list)
print("Accuracy on test data is: ", accuracy(test_list, data_test))


# In[24]:


data_train = [0]*800000 + [4]*800000
print("Accuracy on training data is: ", accuracy((tweet_list0+tweet_list4), data_train))


# In[15]:


#Random Predictor:

# seed the pseudorandom number generator
from random import random

def random_predictor_accuracy(X, Y):
    accuracy = 0.0
    for i in range(0, len(X)):
        predict = random()
        if((predict>=0.5 and Y[i]==4) or (predict<0.5 and Y[i]==0)):
            accuracy += 1.0
    return accuracy/(len(X))


# In[18]:


print(random_predictor_accuracy(test_list, data_test))


# In[21]:


def majority_predictor_accuracy(X, Y):
    class1 = 0
    class2 = 0
    for i in range(0, len(X)):
        if(Y[i]==0):
            class1 +=1
        else:
            class2 +=1
    print(class1)
    print(class2)
    
    return max(class1, class2)/(len(X))


# In[22]:


print(majority_predictor_accuracy(test_list, data_test))

