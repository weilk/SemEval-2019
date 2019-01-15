#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 

from sklearn.feature_extraction.text import CountVectorizer
from keras.layers import Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras import optimizers
import tensorflow as tf


import keras.backend as K


# In[2]:


def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


# In[3]:


from utils import *
df = functions.parse_file(r"raw_data/EmoContext/train.txt", "EmoContext")


# #### Short X Construction

# In[4]:


text_data = []
for idx,row in df.iterrows():
    text_data.append("{}. {}. {}.".format(row['turn1'], row['turn2'], row['turn3']))


# In[6]:


NR_WORDS = 5000


# In[7]:


tokenizer = Tokenizer(num_words=NR_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                   lower=True,split=' ')

tokenizer.fit_on_texts(text_data)


# In[8]:


#tokenizer.word_index


# In[9]:


X = tokenizer.texts_to_sequences(text_data)
X = pad_sequences(X)


# #### Y Construction

# In[10]:


words = {"others": 0, "angry": 1, "sad":2, "happy": 3}

def one_hot_vector(word):
    y = [0,0,0,0]
    y[words[word]] = 1
    return y

Y = []

for idx,row in df.iterrows():
    Y.append(one_hot_vector(row['label']))

Y = np.array(Y)


# #### Model construction

# In[11]:


embed_dim = 128
lstm_out = 32
batch_size = 128

model = Sequential()
model.add(Embedding(NR_WORDS, embed_dim,input_length = X.shape[1]))
model.add(Dropout(0.3))
model.add(LSTM(lstm_out, dropout=0.5))
model.add(Dense(4,activation='softmax'))
adam = optimizers.Adam(lr=0.01)
rmsprop = optimizers.RMSprop(lr=0.005, rho=0.9, epsilon=None, decay=0.0)
model.compile(loss = 'binary_crossentropy', optimizer=rmsprop, metrics = ['accuracy', f1])



# In[13]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0, random_state = 42)


# #### Let's fit

# In[29]:


from sklearn.model_selection import KFold
# prepare cross validation
kfold = KFold(n_splits=6)
Y_train = np.array(Y_train)
# enumerate splits
for train, validation in kfold.split(X_train):
    history = model.fit(X_train[train], Y_train[train],
                    validation_data=(X_train[validation], Y_train[validation]),
                    epochs=3, verbose=1, batch_size=128)

kfold = KFold(n_splits=4)
Y_train = np.array(Y_train)
# enumerate splits
for train, validation in kfold.split(X_train):
    history = model.fit(X_train[train], Y_train[train],
                    validation_data=(X_train[validation], Y_train[validation]),
                    epochs=2, verbose=1, batch_size=128)


# #### Evaluation

# In[ ]:

"""
ev_result = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
print("Score: %.3f" % (ev_result[0]))
print("Validation Accuracy: %.3f" % (ev_result[1]))
print("F1 score: %.3f" % (ev_result[2]))
"""

# In[30]:


model_json = model.to_json()
with open("simple_model.json", "w") as outfile:
    outfile.write(model_json)
model.save_weights("simple_model.h5")


# #### F0r submission

# In[31]:


df_test = functions.parse_file(r"raw_data/EmoContext/devwithoutlabels.txt", "EmoContext")
df_test.head()


# In[32]:


text_data = []

for idx,row in df_test.iterrows():
    text_data.append("{}. {}. {}.".format(row['turn1'], row['turn2'], row['turn3']))
    
    
X_test = tokenizer.texts_to_sequences(text_data)
X_test = pad_sequences(X_test, maxlen = X.shape[1])


# In[33]:


res = model.predict(X_test, batch_size=128, verbose=1)


# In[34]:


revers_words = {0:"others", 1:"angry", 2:"sad", 3:"happy"}

def softmax_convert(res):
    max_i = 0
    max_v = 0
    for i in range(0,4):
        if res[i] > max_v:
            max_v = res[i]
            max_i = i
    return revers_words[max_i]
    
    


# In[35]:


results = []
for r in res:
    results.append(softmax_convert(r))


# In[36]:


df_test['label'] = results


# In[37]:


df_test.head(30)


# In[39]:


df_test.to_csv("simple_model.txt",index=False , sep="\t")


# In[ ]:




