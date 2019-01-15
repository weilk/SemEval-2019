#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 

from sklearn.feature_extraction.text import CountVectorizer
from keras.layers import Dropout, Input, Concatenate,concatenate
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
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
MAX_PROP_LENGTH = 128
tokenizer = Tokenizer(num_words=NR_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                   lower=True,split=' ')

tokenizer.fit_on_texts(text_data)

X_train_1 = tokenizer.texts_to_sequences(df["turn1"])
X_train_1 = pad_sequences(X_train_1, maxlen = MAX_PROP_LENGTH)

X_train_2 = tokenizer.texts_to_sequences(df["turn2"])
X_train_2 = pad_sequences(X_train_2, maxlen = MAX_PROP_LENGTH)

X_train_3 = tokenizer.texts_to_sequences(df["turn3"])
X_train_3 = pad_sequences(X_train_3, maxlen = MAX_PROP_LENGTH)


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

Y_train = []

for idx,row in df.iterrows():
    Y_train.append(one_hot_vector(row['label']))

Y_train = np.array(Y_train)


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

X_test_1 = tokenizer.texts_to_sequences(df["turn1"])
X_test_1 = pad_sequences(X_test_1, maxlen = MAX_PROP_LENGTH)

X_test_2 = tokenizer.texts_to_sequences(df["turn2"])
X_test_2 = pad_sequences(X_test_2, maxlen = MAX_PROP_LENGTH)

X_test_3 = tokenizer.texts_to_sequences(df["turn3"])
X_test_3 = pad_sequences(X_test_3, maxlen = MAX_PROP_LENGTH)


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

Y_test = []

for idx,row in df.iterrows():
    Y_test.append(one_hot_vector(row['label']))

Y_test = np.array(Y_test)


# In[11]:


embed_dim = 128
lstm_out = 64
batch_size = 32

adam = optimizers.Adam(lr=0.01)
rmsprop = optimizers.RMSprop(lr=0.005, rho=0.9, epsilon=None, decay=0.0)

I_1 = Input(shape=X_train_1.shape[1:])
O_1 = Embedding(NR_WORDS, embed_dim,input_length = X_train_1.shape[1])(I_1)
O_1 = LSTM(lstm_out, dropout=0.5)(O_1)

I_2 = Input(shape=X_train_2.shape[1:])
O_2 = Embedding(NR_WORDS, embed_dim,input_length = X_train_2.shape[1])(I_2)
O_2 = LSTM(lstm_out, dropout=0.5)(O_2)

I_3 = Input(shape=X_train_3.shape[1:])
O_3 = Embedding(NR_WORDS, embed_dim,input_length = X_train_3.shape[1])(I_3)
O_3 = LSTM(lstm_out, dropout=0.5)(O_3)

result = concatenate([O_1,O_2,O_3])
result = Dense(128)(result)
result = Dropout(0.4)(result)
result = Dense(64)(result)
result = Dropout(0.4)(result)
result = Dense(4,activation='softmax')(result)
model = Model(inputs=[I_1,I_2,I_3], outputs=result)
model.compile(loss = 'categorical_crossentropy', optimizer="adagrad", metrics = ['accuracy', f1])


# In[12]:


model.build(np.array([X_train_1,X_train_2,X_train_3]).shape)
model.summary()


# In[13]:


mdcheck = ModelCheckpoint("trained_models/best_model_val_acc{val_acc:.4f}.h5", monitor='val_f1', save_best_only=True)


# In[14]:


Y_train = np.array(Y_train)
total = len(Y_train)
try:
    history = model.fit([X_train_1,X_train_2,X_train_3], Y_train,
                        validation_data=([X_test_1,X_test_2,X_test_3], Y_test),
                        epochs=20, verbose=1, batch_size=batch_size,
                        class_weight={
                            0: total / len(np.where(Y_train[:,0]==1.0)[0]),
                            1: total / len(np.where(Y_train[:,1]==1.0)[0]),
                            2: total / len(np.where(Y_train[:,2]==1.0)[0]),
                            3: total / len(np.where(Y_train[:,3]==1.0)[0]),
                        },callbacks=[mdcheck])
except KeyboardInterrupt:
    pass


# In[15]:


model_json = model.to_json()
with open("lstm_normal_model.json", "w") as outfile:
    outfile.write(model_json)
model.save_weights("lstm_normal_12_epochs_cv.h5")


# In[16]:


df_test = functions.parse_file(r"raw_data/EmoContext/devwithoutlabels.txt", "EmoContext")
df_test.head()


# In[17]:


res = model.predict([X_test_1,X_test_2,X_test_3], batch_size=64, verbose=1)
res[3]


# In[18]:


revers_words = {0:"others", 1:"angry", 2:"sad", 3:"happy"}

def softmax_convert(res):
    max_i = 0
    max_v = 0
    for i in range(0,4):
        if res[i] > max_v:
            max_v = res[i]
            max_i = i
    return revers_words[max_i]


# In[19]:


results = []
for r in res:
    results.append(softmax_convert(r))
    
df_test['label'] = results
df_test.head(50)
df_test.to_csv("lstm_normal_12_epochs_cv.txt",index=False , sep="\t")

