from classes import model
import pandas as pd
import numpy as np
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
        
        emb_turn1 = sequence.pad_sequences([one_hot(d, vocab_size) for d in D["turn1"]], maxlen=max_length)
        emb_turn2 = sequence.pad_sequences([one_hot(d, vocab_size) for d in D["turn2"]], maxlen=max_length)
        emb_turn3 = sequence.pad_sequences([one_hot(d, vocab_size) for d in D["turn3"]], maxlen=max_length)
        
        D = D.drop(['turn1','turn2','turn3'],axis=1).values
        return self.model.predict([D,emb_turn1,emb_turn2,emb_turn3])
    

    def train(self,D):
        self.labels = D[output_emocontext].values
        D = D.drop(output_emocontext,axis=1)
        vocab_size = 300
        max_length = 200
        embedding_vector_length = 32
        
        emb_turn1 = sequence.pad_sequences([one_hot(d, vocab_size) for d in D["turn1"]], maxlen=max_length)
        emb_turn2 = sequence.pad_sequences([one_hot(d, vocab_size) for d in D["turn2"]], maxlen=max_length)
        emb_turn3 = sequence.pad_sequences([one_hot(d, vocab_size) for d in D["turn3"]], maxlen=max_length)
        
        D = D.drop(['turn1','turn2','turn3'],axis=1).values


        features_input = Input(shape=(D.shape[1],))
        
        emb_input1 = Input(shape=(max_length,))
        word_emb_turn1 = Embedding(vocab_size, embedding_vector_length, input_length=max_length)(emb_input1)

        emb_input2 = Input(shape=(max_length,))
        word_emb_turn2 = Embedding(vocab_size, embedding_vector_length, input_length=max_length)(emb_input2)

        emb_input3 = Input(shape=(max_length,))
        word_emb_turn3 = Embedding(vocab_size, embedding_vector_length, input_length=max_length)(emb_input3)

        concat_emb = concatenate([word_emb_turn1, word_emb_turn2, word_emb_turn3],axis=-1)
        lstm = LSTM(40)(concat_emb)
        concat_input = Concatenate()([features_input,lstm])
        layer1 = Dense(256)(concat_input)
        activation1 = Activation("relu")(layer1)
        layer2 = Dense(len(output_emocontext))(activation1)
        activation2 = Activation("softmax")(layer2)

        self.model = Model([features_input,emb_input1,emb_input2,emb_input3],activation2)

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
        adagrad = Adagrad(lr=0.001, epsilon=None, decay=0.0)
        adadelta = Adadelta(lr=1.0, rho=0.98, epsilon=None, decay=0.0)

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=adagrad,
            metrics=['accuracy']
        )
        self.model.fit([D,emb_turn1,emb_turn2,emb_turn3], self.labels, epochs=1, batch_size=32)
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

        print(self.model.evaluate([D,emb_turn1,emb_turn2,emb_turn3], self.labels, batch_size=32))
        print("Done testing")