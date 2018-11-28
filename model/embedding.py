from classes import model
import pandas as pd
import numpy as np
from keras.models import load_model
from keras.models import Sequential, Model
from keras.layers import *
from keras.optimizers import SGD, Adam, Adagrad, Adadelta 
from keras.preprocessing import sequence 
from keras.preprocessing.text import one_hot 
from keras.initializers import Constant
from keras.callbacks import EarlyStopping, ModelCheckpoint
from utils import *

class embedding(model):
    
    def forward_pass(self,D):
        emb1_input = np.array([np.array(x) for x in D['embedding_200_turn1'].values])
        emb2_input = np.array([np.array(x) for x in D['embedding_200_turn2'].values])
        emb3_input = np.array([np.array(x) for x in D['embedding_200_turn3'].values])
        D = D.drop(['embedding_200_turn1', 'embedding_200_turn2', 'embedding_200_turn3'], axis=1)
        return self.model.predict([D,emb1_input,emb2_input,emb3_input])
    

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

        x = concatenate([features, lstm1_out, lstm2_out, lstm3_out])
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(128, activation='relu')(x)

        main_out = Dense(self.labels.shape[1], activation='softmax', name='output')(x)
        self.model = Model(inputs=[features, input_turn1, input_turn2, input_turn3], outputs=[main_out])
        # self.model.add(SpatialDropout1D(0.2))
        # self.model.add(Bidirectional(CuDNNLSTM(128, return_sequences=True)))
        # self.model.add(Bidirectional(CuDNNLSTM(64)))
        # self.model.add(Dropout(0.25))
        # self.model.add(Dense(units=5, activation='softmax'))
        
        # filepath="TAIP/SemEval-2019/trained_models/"+self._name+"weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
        loaded = False
        filepath="TAIP/SemEval-2019/trained_models/" + self._name + ".model"
        if os.path.isfile(filepath):
            loaded = True
            self.model.load_weights(filepath)
            print("Loaded model")
        else:
            checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        


        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
        print(self.model.summary())
        if not loaded:
            self.model.fit([D, emb1_input, emb2_input, emb3_input],
                           self.labels,
                           epochs=1,
                           batch_size=1280,
                           validation_split=0.2,
                           callbacks=[EarlyStopping(patience=2), checkpoint])
            print("Done training")

    def test(self,D):
        #TODO
        pass
