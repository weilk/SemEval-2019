#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 

from sklearn.feature_extraction.text import CountVectorizer
from keras.layers import Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential,clone_model
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


# In[5]:


NR_WORDS = 5000
MAX_PROP_LENGTH = 200
tokenizer = Tokenizer(num_words=NR_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                   lower=True,split=' ')

tokenizer.fit_on_texts(text_data)

X_train = tokenizer.texts_to_sequences(text_data)
X_train = pad_sequences(X_train, maxlen = MAX_PROP_LENGTH)


# In[6]:


def one_hot_vector(word,label=None):
    words = {"others": 0, "angry": 1, "sad":2, "happy": 3}
    if label == None:
        y = [0,0,0,0]
        y[words[word]] = 1
        return y
    if label == word:
        return [1,0]
    return [0,1]

Y_train_others = []
Y_train_angry = []
Y_train_sad = []
Y_train_happy = []

for idx,row in df.iterrows():
    Y_train_others.append(one_hot_vector(row['label'],"others"))
    Y_train_angry.append(one_hot_vector(row['label'],"angry"))
    Y_train_sad.append(one_hot_vector(row['label'],"sad"))
    Y_train_happy.append(one_hot_vector(row['label'],"happy"))

Y_train_others = np.array(Y_train_others)
Y_train_angry = np.array(Y_train_angry)
Y_train_sad = np.array(Y_train_sad)
Y_train_happy = np.array(Y_train_happy)


# In[7]:


df = functions.parse_file(r"raw_data/EmoContext/devwithlabels.txt", "EmoContext")
df.head(5)


# In[8]:


text_data = []
for idx,row in df.iterrows():
    text_data.append("{}. {}. {}.".format(row['turn1'], row['turn2'], row['turn3']))


# In[9]:


tokenizer = Tokenizer(num_words=NR_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                   lower=True,split=' ')

tokenizer.fit_on_texts(text_data)

X_test = tokenizer.texts_to_sequences(text_data)
X_test = pad_sequences(X_test, maxlen = MAX_PROP_LENGTH)


# In[10]:


def one_hot_vector(word,label=None):
    words = {"others": 0, "angry": 1, "sad":2, "happy": 3}
    if label == None:
        y = [0,0,0,0]
        y[words[word]] = 1
        return y
    if label == word:
        return [1,0]
    return [0,1]

Y_test_others = []
Y_test_angry = []
Y_test_sad = []
Y_test_happy = []

for idx,row in df.iterrows():
    Y_test_others.append(one_hot_vector(row['label'],"others"))
    Y_test_angry.append(one_hot_vector(row['label'],"angry"))
    Y_test_sad.append(one_hot_vector(row['label'],"sad"))
    Y_test_happy.append(one_hot_vector(row['label'],"happy"))

Y_test_others = np.array(Y_test_others)
Y_test_angry = np.array(Y_test_angry)
Y_test_sad = np.array(Y_test_sad)
Y_test_happy = np.array(Y_test_happy)


# In[11]:


embed_dim = 128
lstm_out = 32
batch_size = 64

adam = optimizers.Adam(lr=0.01)
rmsprop = optimizers.RMSprop(lr=0.005, rho=0.9, epsilon=None, decay=0.0)


model = Sequential()
model.add(Embedding(NR_WORDS, embed_dim,input_length = X_train.shape[1]))
model.add(LSTM(lstm_out,dropout=0.5))
model.add(Dense(2,activation='softmax'))


# In[12]:


model.summary()


# In[13]:


mdcheck = ModelCheckpoint("trained_models/best_model_val_acc{val_acc:.4f}.h5", monitor='val_f1', save_best_only=True)


# In[14]:


Y_train_angry = np.array(Y_train_angry)

model_angry = clone_model(model)
model_angry.compile(loss = 'binary_crossentropy', optimizer=rmsprop, metrics = ['accuracy', f1])
history = model_angry.fit(X_train, Y_train_angry,
                    validation_data=(X_test, Y_test_angry),
                    epochs=5, verbose=1, batch_size=batch_size,callbacks=[mdcheck])


# In[15]:


Y_train_sad = np.array(Y_train_sad)

model_sad = clone_model(model)
model_sad.compile(loss = 'binary_crossentropy', optimizer=rmsprop, metrics = ['accuracy', f1])
history = model_sad.fit(X_train, Y_train_sad,
                    validation_data=(X_test, Y_test_sad),
                    epochs=5, verbose=1, batch_size=batch_size,callbacks=[mdcheck])


# In[16]:


Y_train_happy = np.array(Y_train_happy)

model_happy = clone_model(model)
model_happy.compile(loss = 'binary_crossentropy', optimizer=rmsprop, metrics = ['accuracy', f1])
history = model_happy.fit(X_train, Y_train_happy,
                    validation_data=(X_test, Y_test_happy),
                    epochs=5, verbose=1, batch_size=batch_size,callbacks=[mdcheck])


# In[17]:


Y_train_others = np.array(Y_train_others)

model_others = clone_model(model)
model_others.compile(loss = 'binary_crossentropy', optimizer=rmsprop, metrics = ['accuracy', f1])
history = model_others.fit(X_train, Y_train_others,
                    validation_data=(X_test, Y_test_others),
                    epochs=5, verbose=1, batch_size=batch_size,callbacks=[mdcheck])


# In[18]:


model_json = model.to_json()
with open("lstm_normal_model_others.json", "w") as outfile:
    outfile.write(model_json)
model.save_weights("lstm_normal_12_epochs_cv_others.h5")


# In[19]:


model_json = model.to_json()
with open("lstm_normal_model_angry.json", "w") as outfile:
    outfile.write(model_json)
model.save_weights("lstm_normal_12_epochs_cv_angry.h5")


# In[20]:


model_json = model.to_json()
with open("lstm_normal_model_sad.json", "w") as outfile:
    outfile.write(model_json)
model.save_weights("lstm_normal_12_epochs_cv_sad.h5")


# In[21]:


model_json = model.to_json()
with open("lstm_normal_model_happy.json", "w") as outfile:
    outfile.write(model_json)
model.save_weights("lstm_normal_12_epochs_cv_happy.h5")

