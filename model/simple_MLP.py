from classes import model
import pandas as pd
import numpy as np
from keras.models import load_model
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, InputLayer, Embedding, Input, LSTM, Concatenate, concatenate
from keras.optimizers import SGD, Adam, Adagrad, Adadelta 
from keras.preprocessing import sequence 
from keras.preprocessing.text import one_hot 

from utils import *

class simple_MLP(model):
    
    def forward_pass(self,D):
        vocab_size = 300
        max_length = 200
        embedding_vector_length = 32
        
        emb_turn1 = D["embedding_200_turn1"]
        emb_turn2 = D["embedding_200_turn2"]
        emb_turn3 = D["embedding_200_turn3"]
        
        return self.model.predict([D,emb_turn1,emb_turn2,emb_turn3])
    
    def train(self,D):
        self.labels = D[output_emocontext].values # labels
        D = D.drop(output_emocontext,axis=1) # data 
        vocab_size = 300
        embedding_vector_length = 64
        output_dim = 128

        emb_turn1 = D["embedding_200_turn1"][0]
        emb_turn2 = D["embedding_200_turn2"][0]
        emb_turn3 = D["embedding_200_turn3"][0]
        # emb = np.concatenate((emb_turn1, emb_turn2, emb_turn3))

        # max_length = np.shape(emb_turn1)[0] * np.shape(emb_turn1)[1]
        
        print("emb shape")
        print(np.shape(emb_turn1))
        print(np.shape(emb_turn2))
        print(np.shape(emb_turn3))
        print(np.shape(emb_turn3[0]))
        print("shape")
        print(D.shape)
        features_input = Input(shape=(D.shape[1],output_dim))

        emb_input1 = Input(shape=(emb_turn1.shape[1],))
        word_emb_turn1 = Embedding(np.shape(emb_turn1)[1],
                                   output_dim)(emb_input1)

        emb_input2 = Input(shape=(emb_turn2.shape[1],))
        word_emb_turn2 = Embedding(np.shape(emb_turn2)[1],
                                   output_dim)(emb_input2)

        emb_input3 = Input(shape=(emb_turn3.shape[1],))
        word_emb_turn3 = Embedding(np.shape(emb_turn3)[1],
                                   output_dim)(emb_input3)


        # concat_emb = concatenate([emb_input1, emb_input2, emb_input3],axis=-1)
        concat_emb = concatenate([word_emb_turn1, word_emb_turn2, word_emb_turn3],axis=1)
        concat_shape = concat_emb.get_shape()[1:]
        # concat_emb.set_shape(concat_shape)
        print(word_emb_turn1.get_shape())
        print(word_emb_turn2.get_shape())
        print(word_emb_turn3.get_shape())
        print("shape of concat")
        # emb_input = Input(shape=np.shape(emb))
        # embedding_layer = Embedding(np.shape(emb), 64)(emb_input)

        print(concat_emb)
        print(dir(concat_emb))
        print(concat_emb.get_shape())
        lstm = LSTM(output_dim, return_sequences=True)(concat_emb)
        # lstm = LSTM(64, return_sequences=True)(embedding_layer)
        concat_input = concatenate([features_input, lstm], axis=1)

        layer1 = Dense(256)(concat_input)
        activation1 = Activation("relu")(layer1)

        layer2 = Dense(128)(activation1)
        activation2 = Activation("relu")(layer2)

        layer3 = Dense(len(output_emocontext))(activation2)
        activation3 = Activation("softmax")(layer3)

        self.model = Model([features_input, emb_input1,emb_input2,emb_input3],activation3)

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        #adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
        adagrad = Adagrad(lr=0.001, epsilon=None, decay=0.0)
        adadelta = Adadelta(lr=1.0, rho=0.985, epsilon=None, decay=0.0)

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=adadelta,
            metrics=['accuracy']
        )
        total = len(np.where(self.labels[:,0]==1.0)[0])*(1.0-0.2) + len(np.where(self.labels[:,1]==1.0)[0])*(1.0-0.2) + len(np.where(self.labels[:,2]==1.0)[0])*(1.0-0.2) + len(np.where(self.labels[:,3]==1.0)[0])*(1.0-0.2)
        # self.model.fit([D,emb_turn1,emb_turn2,emb_turn3], self.labels, epochs=5, batch_size=128,validation_split=0.2, class_weight={
        #     0: total / len(np.where(self.labels[:,0]==1.0)[0])*(1.0-0.2),
        #     1: total / len(np.where(self.labels[:,1]==1.0)[0])*(1.0-0.2),
        #     2: total / len(np.where(self.labels[:,2]==1.0)[0])*(1.0-0.2),
        #     3: total / len(np.where(self.labels[:,3]==1.0)[0])*(1.0-0.2)
        # })
        # print("Done training")

    def test(self,D):
        self.labels = pd.get_dummies(D[output_emocontext])
        D = D.drop(output_emocontext,axis=1)
        
        vocab_size = 300
        max_length = 200
        embedding_vector_length = 32
        
        emb_turn1 = D["embedding_200_turn1"]
        emb_turn2 = D["embedding_200_turn2"]
        emb_turn3 = D["embedding_200_turn3"]

        results = self.model.evaluate([D,emb_turn1,emb_turn2,emb_turn3], self.labels, batch_size=32)
        print(results)
        print("Done testing")
        return results

    def test_diff(self,data, labels):
        self.labels = labels
        D = data

        vocab_size = 300
        max_length = 200
        embedding_vector_length = 32
        
        emb_turn1 = D["embedding_200_turn1"]
        emb_turn2 = D["embedding_200_turn2"]
        emb_turn3 = D["embedding_200_turn3"]

        results = self.model.evaluate([D,emb_turn1,emb_turn2,emb_turn3], self.labels, batch_size=32)
        print(results)
        print("Done testing")
        return results