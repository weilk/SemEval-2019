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
from keras.callbacks import ModelCheckpoint
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
df.head(5)


# In[4]:


text_data = []
for idx,row in df.iterrows():
    text_data.append("{}. {}. {}.".format(row['turn1'], row['turn2'], row['turn3']))


# In[ ]:


NR_WORDS = 5000
MAX_PROP_LENGTH = 200
tokenizer = Tokenizer(num_words=NR_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                   lower=True,split=' ')

tokenizer.fit_on_texts(text_data)

X_train = tokenizer.texts_to_sequences(text_data)
X_train = pad_sequences(X_train, maxlen = MAX_PROP_LENGTH)


# In[ ]:


def one_hot_vector(word,label=None):
    words = {"others": 0, "angry": 1, "sad":2, "happy": 3}
    if label == None:
        y = [0,0,0,0]
        y[words[word]] = 1
        return y
    if label == word:
        return [1,0]
    return [0,1]

Y_train = []

for idx,row in df.iterrows():
    Y_train.append(one_hot_vector(row['label']))

Y_train = np.array(Y_train)


# In[ ]:


df = functions.parse_file(r"raw_data/EmoContext/devwithlabels.txt", "EmoContext")
df.head(5)


# In[ ]:


text_data = []
for idx,row in df.iterrows():
    text_data.append("{}. {}. {}.".format(row['turn1'], row['turn2'], row['turn3']))


# In[ ]:


tokenizer = Tokenizer(num_words=NR_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                   lower=True,split=' ')

tokenizer.fit_on_texts(text_data)

X_test = tokenizer.texts_to_sequences(text_data)
X_test = pad_sequences(X_test, maxlen = MAX_PROP_LENGTH)


# In[ ]:


def one_hot_vector(word,label=None):
    words = {"others": 0, "angry": 1, "sad":2, "happy": 3}
    if label == None:
        y = [0,0,0,0]
        y[words[word]] = 1
        return y
    if label == word:
        return [1,0]
    return [0,1]

Y_test = []

for idx,row in df.iterrows():
    Y_test.append(one_hot_vector(row['label']))

Y_test = np.array(Y_test)


# In[ ]:


embed_dim = 128
lstm_out = 128
batch_size = 128

adam = optimizers.Adam(lr=0.01)
rmsprop = optimizers.RMSprop(lr=0.005, rho=0.9, epsilon=None, decay=0.0)


model = Sequential()
model.add(Embedding(NR_WORDS, embed_dim,input_length = X_train.shape[1]))
model.add(LSTM(lstm_out))
model.add(Dense(128))
model.add(Dropout(0.3))
model.add(Dense(4,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer=rmsprop, metrics = ['accuracy', f1])


# In[ ]:


model.summary()


# In[ ]:


mdcheck = ModelCheckpoint("trained_models/best_model_val_acc{val_acc:.4f}.h5", monitor='val_acc', save_best_only=True)


# In[ ]:


Y_train = np.array(Y_train)
total = len(Y_train)
history = model.fit(X_train, Y_train,
                    validation_data=(X_test, Y_test),
                    epochs=20, verbose=1, batch_size=batch_size,
                    class_weight={
                        0: total / len(np.where(Y_train[:,0]==1.0)[0]),
                        1: total / len(np.where(Y_train[:,1]==1.0)[0]),
                        2: total / len(np.where(Y_train[:,2]==1.0)[0]),
                        3: total / len(np.where(Y_train[:,3]==1.0)[0]),
                    },callbacks=[mdcheck])


# In[ ]:


model_json = model.to_json()
with open("lstm_normal_model.json", "w") as outfile:
    outfile.write(model_json)
model.save_weights("lstm_normal_12_epochs_cv.h5")


# In[ ]:


df_test = functions.parse_file(r"raw_data/EmoContext/devwithoutlabels.txt", "EmoContext")
df_test.head()


# In[ ]:


text_data = []

for idx,row in df_test.iterrows():
    text_data.append("{}. {}. {}.".format(row['turn1'], row['turn2'], row['turn3']))

res = model.predict(X_test, batch_size=64, verbose=1)
res[3]


# In[ ]:


revers_words = {0:"others", 1:"angry", 2:"sad", 3:"happy"}

def softmax_convert(res):
    max_i = 0
    max_v = 0
    for i in range(0,4):
        if res[i] > max_v:
            max_v = res[i]
            max_i = i
    return revers_words[max_i]


# In[ ]:


results = []
for r in res:
    results.append(softmax_convert(r))
    
df_test['label'] = results
df_test.head(50)
df_test.to_csv("lstm_normal_12_epochs_cv.txt",index=False , sep="\t")

