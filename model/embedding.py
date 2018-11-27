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
    

    def add_turn_layer(self, index):
        input_turn_length = len(self.data['embedding_200_turn%s' % index][0])
        input_turn = Input(shape=(input_turn_length,), name='turn%s_input' % index)
        emb = Embedding(input_dim=np.shape(self.embedding_matrix)[0],
                            output_dim=self.embedding_dim,
                            embeddings_initializer=Constant(self.embedding_matrix),
                            input_length=input_turn_length,
                            trainable=True)(input_turn)
        return input_turn, LSTM(128)(emb)

    def _input_turn_embeddings(self, D):
        emb1_input = np.array([np.array(x) for x in D['embedding_200_turn1'].values])
        emb2_input = np.array([np.array(x) for x in D['embedding_200_turn2'].values])
        emb3_input = np.array([np.array(x) for x in D['embedding_200_turn3'].values])
        return [emb1_input, emb2_input, emb3_input]

    def train(self, D, embedding_matrix, embedding_dim=200):
        self.embedding_matrix = embedding_matrix
        self.embedding_dim = embedding_dim

        self.labels = D[output_emocontext].values
        self.data = D
        D = D.drop(output_emocontext,axis=1) # data 
        emb1_input = np.array([np.array(x) for x in D['embedding_200_turn1'].values])
        emb2_input = np.array([np.array(x) for x in D['embedding_200_turn2'].values])
        emb3_input = np.array([np.array(x) for x in D['embedding_200_turn3'].values])
        
        print("Dropping emb cols: " )
        D = D.drop(['embedding_200_turn1', 'embedding_200_turn2', 'embedding_200_turn3'], axis=1)

        self.model = Sequential()

        features = Input(shape=(D.shape[1],), name="features_input")
        input_turn1, lstm1_out = self.add_turn_layer("1")
        input_turn2, lstm2_out = self.add_turn_layer("2")
        input_turn3, lstm3_out = self.add_turn_layer("3")

        # features = Input(shape = )

        x = concatenate([lstm1_out, lstm2_out, lstm3_out])
        x = Dense(200, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(64, activation='relu')(x)

        main_out = Dense(self.labels.shape[1], activation='softmax', name='output')(x)
        self.model = Model(inputs=[features, input_turn1, input_turn2, input_turn3], outputs=[main_out])
        # self.model.add(SpatialDropout1D(0.2))
        # self.model.add(Bidirectional(CuDNNLSTM(128, return_sequences=True)))
        # self.model.add(Bidirectional(CuDNNLSTM(64)))
        # self.model.add(Dropout(0.25))
        # self.model.add(Dense(units=5, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
        print(self.model.summary())

        self.model.fit([D, emb1_input, emb2_input, emb3_input], self.labels, epochs=10, batch_size=128,validation_split=0.2)
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
