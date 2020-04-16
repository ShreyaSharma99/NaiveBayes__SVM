from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from time import time

from sklearn.feature_selection import SelectPercentile, f_classif


#reading data function 
def read_data_train(file_name):
    f = pd.read_csv(file_name, header = None,  encoding='ISO-8859-1') 
    f = f.to_numpy() #this is a 1600000*6 numpy array now
    
    tweet_list0 = []
    tweet_list4 = []
    data = []
    for tweet in f:
        if (tweet[0]==0):
            tweet_list0.append(tweet[5])
        elif (tweet[0]==4):
            tweet_list4.append(tweet[5])
    return tweet_list0, tweet_list4
        
def read_data_test(file_name):
    f = pd.read_csv(file_name, header = None,  encoding='ISO-8859-1') 
    f = f.to_numpy() #this is a 1600000*6 numpy array now
    
    tweet_list = []
    data = []
    for tweet in f:
        if (tweet[0]!=2):
            tweet_list.append(tweet[5])
            data.append(tweet[0])
    return tweet_list, data


# In[3]:


#reading data from the training set of 1600000 samples
tweet_list0, tweet_list4 = read_data_train('/home/shreya/Sem6/COL774/Assgn2/trainingandtestdata/training.1600000.processed.noemoticon.csv')
train_label = ([0]*800000) + ([4]*800000)


#reading data from the testing set of 498 samples
test_list, test_label = read_data_test('/home/shreya/Sem6/COL774/Assgn2/trainingandtestdata/testdata.manual.2009.06.14.csv')
print(len(test_list))


# In[5]:


print(tweet_list0[0])
print(tweet_list4[0])
print(test_list[0])


# In[6]:


vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, min_df = 0.001, stop_words='english', smooth_idf = True)
X = vectorizer.fit_transform(tweet_list0 + tweet_list4)


X_test = vectorizer.transform(test_list)


t0 = time()
N = 1600000
batch_size = 1000
r = 1600  #number of batches
Y_label = ([0]*1000)

model = GaussianNB()
# features_train = X.toarray()
# model.fit(X.toarray(), [0,4])
for i in range(0, r):
#     print(i)
    if (i>=800):
        Y_label = ([4]*1000)
#     features_train = get_features(X, i, batch_size)
    model.partial_fit(X[batch_size*i:batch_size*(i+1)].toarray(), Y_label, [0, 4])

print(f'\nTraining time: {round(time()-t0, 3)}s')


# In[11]:


t0 = time()
print(f'Prediction time (train): {round(time()-t0, 3)}s')
t0 = time()
# X_test = vectorizer.transform(test_list)
score_test = model.score(X_test.toarray(), test_label)
print(f'Prediction time (test): {round(time()-t0, 3)}s')
# print('Train set score:', score_train)
print('Test set score:', score_test)


##########################################################################ROC_CURVE###############################################################################

print(model.predict((X_test.toarray())))

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve for Y_Label = 0')
    plt.legend()
    plt.show()


# In[23]:



y_test = np.zeros((len(test_label), 2))
y_score = np.zeros((len(test_label), 2))

y_predict = model.predict((X_test.toarray()))

x_test = X_test.toarray()

for i in range(len(test_label)):
    if test_label[i]==0:
        y_test[i][0] = 1
    else:
        y_test[i][1] = 1
    if y_predict[i] == 0:
        y_score[i][0] = 1
    else:
        y_score[i][1] = 1
        
    
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(0,2):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

auc = roc_auc_score(y_test[:,0], y_score[:,0])
print('AUC: %.2f' % auc)

fpr1, tpr1, thresholds = roc_curve(y_test[:,1], y_score[:,1])
plot_roc_curve(fpr1, tpr1)


# In[25]:


fpr0, tpr0, thresholds = roc_curve(y_test[:,0], y_score[:,0])
plot_roc_curve(fpr0, tpr0)

