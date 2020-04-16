from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import re
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk import ngrams


# from nltk import ngrams

# sentence = 'this is a foo bar sentences and i want to ngramize it'

# n = 2
# sixgrams = ngrams(sentence.split(), 2)
# print((sixgrams))
# list1 = []
# for grams in sixgrams:
# #   print((grams)) 
#     list1.append(list(grams))
# print(list1)


# In[3]:


porter=PorterStemmer()

def stemSentence(token_words):
#     token_words=word_tokenize(sentence)
    token_words
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
    return stem_sentence


# In[4]:


stop_words = pd.read_csv('/home/shreya/Sem6/COL774/Assgn2/trainingandtestdata/stopwords.csv', header = None) 
stop_words = stop_words.to_numpy()
print(stop_words.shape)


# In[20]:


def get_bigram(str_list):
    i = 0
    j = 0
    bi_list = []
    uni_list = []
    while i<len(str_list):
        if (str_list[i] in stop_words) or (len(str_list[i])==0) or (str_list[i][0]=='@'):
            popped = str_list.pop(i)
        else:
            i +=1
    i = 0
    for i in range(0, len(str_list)-1):
        l = [str_list[i], str_list[i+1]]
        bi_list.append(l)
    
    for i in range(0, len(str_list)):
        uni_list.append(str_list[i])
    return bi_list, uni_list


# In[6]:


def get_trigram(str_list):
    i = 0
    j = 0
    bi_list = []
    while i<len(str_list):
        if (str_list[i] in stop_words) or (len(str_list[i])==0) or (str_list[i][0]=='@'):
            popped = str_list.pop(i)
        else:
            i +=1
    i = 0
    for i in range(0, len(str_list)-2):
        l = [str_list[i], str_list[i+1], str_list[i+2]]
        bi_list.append(l)
      
    return bi_list


# In[19]:


print(get_bigram(['hi', 'bro', 'are', 'wassup', '', 'yourf']))


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
            string = re.split(r'[\s,.""''*]+',tweet[5])
            if (tweet[0]==0):
                string = stemSentence(string)
                bi_list, uni_list = get_bigram(string)
#                 tri_list = get_trigram(string)
                tweet_list0.append(uni_list + bi_list)
            elif (tweet[0]==4):
                string = stemSentence(string)
                bi_list, uni_list = get_bigram(string)
#                 tri_list = get_trigram(string)
                tweet_list4.append(uni_list + bi_list)
        return tweet_list0, tweet_list4
    else:
        tweet_list = []
        data = []
        for tweet in f:
            string = re.split(r'[\s,.""''*]+',tweet[5])
#             \s,."*-:/
            if (tweet[0]!=2):
                string = stemSentence(string)
                bi_list, uni_list = get_bigram(string)
#                 tri_list = get_trigram(string)
                tweet_list.append(uni_list + bi_list)
                data.append(tweet[0])
        return tweet_list, data

#reading data from the training set of 1600000 samples
tweet_list0, tweet_list4 = read_data('/home/shreya/Sem6/COL774/Assgn2/trainingandtestdata/training.1600000.processed.noemoticon.csv', True)
print(len(tweet_list0))

#reading data from the testing set of 498 samples
test_list, data_test = read_data('/home/shreya/Sem6/COL774/Assgn2/trainingandtestdata/testdata.manual.2009.06.14.csv', False)
print(len(test_list))

train_size0 = len(tweet_list0)
train_size4 = len(tweet_list4)

test_size = len(test_list)


# In[24]:


vocab = set()

for tweet in (tweet_list0+tweet_list4):
    for w in tweet:
        vocab.add(tuple(w))
#         print(w)


# In[26]:


print(len(vocab))


# In[27]:


Dict = {}
i = 0
for word in vocab:
    Dict[word] = i
    i +=1


# In[28]:


# 0 = negative, 2 = neutral, 4 = positive

p_y0 = float(train_size0)/(train_size0 + train_size4)
p_y4 = float(train_size4)/(train_size0 + train_size4)

print(p_y0)
print(p_y4)


total_words_y0 = 0
total_words_y4 = 0

v = len(vocab)
p_x_y0 = np.ones(v)
p_x_y4 = np.ones(v)

index = 0

for tweet in tweet_list0:
    for word in list(tweet):
#         print(tweet[i])
        index = Dict.get(tuple(word))
        if index!=None:
            p_x_y0[index] +=1

for tweet in tweet_list4:
    for word in list(tweet):
        index = Dict.get(tuple(word))
        if index!=None:
            p_x_y4[index] +=1
print()
p_x_y0 = p_x_y0/(train_size0 + v)
p_x_y4 = p_x_y4/(train_size4 + v)


# In[30]:


# print(Dict.get(tweet_list0[0][1]))
print(p_x_y0[1])


# In[31]:


def test_MLA(tweet):
    p0 = np.log(p_y0)
    p4 = np.log(p_y4)    
    index = 0
    v_len = v
    
#     print(list(tweet))
    for word in list(tweet):
        index = Dict.get(tuple(word))
#         print(index)
        if (index == None):
#             print("here")
            v_len +=1
            p0 += np.log(1.0/(total_words_y0 + v))
            p4 += np.log(1.0/(total_words_y4 + v))
        else:
#             print(index)
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


# In[32]:


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

# print(test_list[0])
print("Accuracy: ",accuracy(test_list, data_test))

