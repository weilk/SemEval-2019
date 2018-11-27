from classes import model
import pandas as pd
import numpy as np
from keras.models import load_model
from keras.models import Sequential, Model
# from keras.layers import Dense, Activation, Dropout, InputLayer, Embedding, Input, LSTM, Concatenate, concatenate
from keras.layers import *
from keras.optimizers import SGD, Adam, Adagrad, Adadelta 
from keras.preprocessing import sequence 
from keras.preprocessing.text import one_hot 
from keras.initializers import Constant

from utils import *

class embedding(model):
    
    def forward_pass(self,D):
        vocab_size = 300
        max_length = 200
        embedding_vector_length = 32
        
        emb_turn1 = sequence.pad_sequences([one_hot(d, vocab_size) for d in D["turn1"]], maxlen=max_length)
        emb_turn2 = sequence.pad_sequences([one_hot(d, vocab_size) for d in D["turn2"]], maxlen=max_length)
        emb_turn3 = sequence.pad_sequences([one_hot(d, vocab_size) for d in D["turn3"]], maxlen=max_length)
        
        D = D.drop(['turn1','turn2','turn3'],axis=1).values
        return self.model.predict([D,emb_turn1,emb_turn2,emb_turn3])
    
    def train(self, D, embedding_matrix, embedding_dim=200):

        self.labels = D[output_emocontext].values

        D = D.drop(output_emocontext,axis=1) # data 
        print(D.shape)
        print(D.columns)


        self.model = Sequential()
        input_turn1_length = len(D['embedding_200_turn1'][0])
        print(input_turn1_length)
        input_turn1 = Input(shape=(input_turn1_length,), name='turn1_input')
        emb1 = Embedding(input_dim=np.shape(embedding_matrix)[0],
                            output_dim=embedding_dim,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=input_turn1_length,
                            trainable=True)(input_turn1)
        lstm1_out = LSTM(64)(emb1)
        main_out = Dense(self.labels.shape[1], activation='softmax', name='output')(lstm1_out)
        self.model = Model(inputs=[input_turn1], outputs=[main_out])
        # self.model.add(SpatialDropout1D(0.2))
        # self.model.add(Bidirectional(CuDNNLSTM(128, return_sequences=True)))
        # self.model.add(Bidirectional(CuDNNLSTM(64)))
        # self.model.add(Dropout(0.25))
        # self.model.add(Dense(units=5, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
        print(self.model.summary())
        print(D.shape)
        emb1_input = np.array([np.array(x) for x in D['embedding_200_turn1'].values])
        print(np.shape(emb1_input))
        # import ipdb
        # ipdb.set_trace(context=10)
        # print(D['embedding_200_turn1'])
        # print(type(D['embedding_200_turn1']))
        # D['embedding_200_turn1'] = D['embedding_200_turn1'].values
        # print(type(D['embedding_200_turn1']))
        # print(D.shape)
        # print(D['embedding_200_turn1'][0])
        # print(self.labels.shape)
        # emb1_input = np.reshape(D['embedding_200_turn1'],(D.shape[0], input_turn1_length))
        
        self.model.fit(emb1_input, self.labels, epochs=10, batch_size=128,validation_split=0.2)
        print("Done training")

    def test(self,D):
        self.labels = pd.get_dummies(D[output_emocontext])
        D = D.drop(output_emocontext,axis=1)
        
        vocab_size = 300
        max_length = 200
        embedding_vector_length = 32
        
        emb_turn1 = sequence.pad_sequences([one_hot(d, vocab_size) for d in D["turn1"]], maxlen=max_length)
        emb_turn2 = sequence.pad_sequences([one_hot(d, vocab_size) for d in D["turn2"]], maxlen=max_length)
        emb_turn3 = sequence.pad_sequences([one_hot(d, vocab_size) for d in D["turn3"]], maxlen=max_length)
        
        D = D.drop(['turn1','turn2','turn3'],axis=1).values

        results = self.model.evaluate([D,emb_turn1,emb_turn2,emb_turn3], self.labels, batch_size=32)
        print(results)
        print("Done testing")
        return results
