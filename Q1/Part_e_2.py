from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import re
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer

porter=PorterStemmer()

def stemSentence(token_words):
#     token_words=word_tokenize(sentence)
    token_words
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
    return stem_sentence


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
                tweet_list0.append(string)
            elif (tweet[0]==4):
                string = stemSentence(string)
                tweet_list4.append(string)
        return tweet_list0, tweet_list4
    else:
        tweet_list = []
        data = []
        for tweet in f:
            string = re.split(r'[\s,.""''*]+',tweet[5])
#             \s,."*-:/
            if (tweet[0]!=2):
                string = stemSentence(string)
                tweet_list.append(string)
                data.append(tweet[0])
        return tweet_list, data


#reading data from the training set of 1600000 samples
tweet_list0, tweet_list4 = read_data('/home/shreya/Sem6/COL774/Assgn2/trainingandtestdata/training.1600000.processed.noemoticon.csv', True)
print(len(tweet_list0))


# In[7]:


#reading data from the testing set of 498 samples
test_list, data_test = read_data('/home/shreya/Sem6/COL774/Assgn2/trainingandtestdata/testdata.manual.2009.06.14.csv', False)
print(len(test_list))

stop_words = pd.read_csv('/home/shreya/Sem6/COL774/Assgn2/trainingandtestdata/stopwords.csv', header = None) 
stop_words = stop_words.to_numpy()
print(stop_words.shape)


# In[10]:


train_size0 = len(tweet_list0)
train_size4 = len(tweet_list4)

test_size = len(test_list)


# In[11]:


vocab = set()


# In[12]:


for tweet in (tweet_list0+tweet_list4):
    line = []
    for w in tweet:
        if len(w)>0:
            if w[0]!="@":  #Remove Twitter Handles
                if w not in stop_words:   #Remove stopwords
                    line.append(w) 
                    vocab.add(w)
    tweet = line


Dict = {}
i = 0
for word in vocab:
    Dict[word] = i
    i +=1

# 0 = negative, 2 = neutral, 4 = positive

p_y0 = float(train_size0)/(train_size0 + train_size4)
p_y4 = float(train_size4)/(train_size0 + train_size4)

print(p_y0)
print(p_y4)


# In[17]:


total_words_y0 = 0
total_words_y4 = 0

v = len(vocab)
p_x_y0 = np.ones(v)
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


# In[18]:


print(Dict.get(tweet_list0[0][1]))
print(p_x_y0[2])


# In[19]:


def test_MLA(tweet):
    p0 = np.log(p_y0)
    p4 = np.log(p_y4)    
    index = 0
    v_len = v
    count0 = 1
    count4 = 1
    for word in tweet:
        index = Dict.get(word)
        if (index == None):
            v_len +=1
            p0 += np.log(1.0/(total_words_y0 + v))
            p4 += np.log(1.0/(total_words_y4 + v))
        else:
            if (p_x_y0[index] > p_x_y4[index]):
                count0 += 1
            else:
                count4 += 1
            p0 += np.log(p_x_y0[index])
            p4 += np.log(p_x_y4[index])
    
    p0 = p0 + np.log(count0/count4)
    p4 = p4 + np.log(count4/count0)
    
#     print("p0: ", p0)
#     print("p4: ", p4)
    
    if(p0>p4):
        return 0
    else:
        return 4


# In[20]:


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


# In[21]:


def remove_stopwords(tweet_list):
    new_list = []
    for tweet in tweet_list:
        line = []
        for w in tweet:
            if len(w)>0:
                if w[0]!="@":
                    if w not in stop_words:
                        line.append(w)
        new_list.append(line)
    return new_list


# In[22]:


print(test_list[0])
test_list_new = remove_stopwords(test_list)
print(test_list_new[0])
print(accuracy(test_list_new, data_test))

