from classes import model
import pandas as pd
import numpy as np
from time import time
from keras.models import load_model
from keras.models import Sequential, Model
from keras.layers import *
from keras.optimizers import SGD, Adam, Adagrad, Adadelta 
from keras.preprocessing import sequence 
from keras.preprocessing.text import one_hot 
from keras.initializers import Constant
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from utils import *

class embedding(model):
    
    def forward_pass(self,D):
        emb1_input = np.array([np.array(x) for x in D['embedding_200_turn1'].values])
        emb2_input = np.array([np.array(x) for x in D['embedding_200_turn2'].values])
        emb3_input = np.array([np.array(x) for x in D['embedding_200_turn3'].values])
        
        D = D.drop(['turn1', 'turn2', 'turn3'], axis=1)
        D = D.drop(['embedding_200_turn1', 'embedding_200_turn2', 'embedding_200_turn3'], axis=1)
        return self.model.predict([D])
    

    def add_turn_layer(self, index):
        input_turn_length = len(self.data['embedding_200_turn%s' % index][0])
        input_turn = Input(shape=(input_turn_length,), name='turn%s_input' % index)
        emb = Embedding(input_dim=np.shape(self.embedding_matrix)[0],
                            output_dim=self.embedding_dim,
                            embeddings_initializer=Constant(self.embedding_matrix),
                            input_length=input_turn_length,
                            trainable=True)(input_turn)
        return input_turn, LSTM(64, activation='relu')(LSTM(64, activation='relu',return_sequences=True)(LSTM(64, activation='relu',return_sequences=True)(LSTM(64, activation='relu',return_sequences=True)(emb))))

    def _input_turn_embeddings(self, D):
        emb1_input = np.array([np.array(x) for x in D['embedding_200_turn1'].values])
        emb2_input = np.array([np.array(x) for x in D['embedding_200_turn2'].values])
        emb3_input = np.array([np.array(x) for x in D['embedding_200_turn3'].values])
        return [emb1_input, emb2_input, emb3_input]

    def train(self, D,trainIdx,validationIdx, embedding_matrix, embedding_dim=200):
        self.embedding_matrix = embedding_matrix
        self.embedding_dim = embedding_dim

        self.labels = D[output_emocontext].values
        D = D.drop(output_emocontext,axis=1) # data 
        self.data = D
        emb1_input = np.array([np.array(x) for x in D['embedding_200_turn1'].values])
        emb2_input = np.array([np.array(x) for x in D['embedding_200_turn2'].values])
        emb3_input = np.array([np.array(x) for x in D['embedding_200_turn3'].values])
        
        print("Dropping emb cols: " )
        D = D.drop(['embedding_200_turn1', 'embedding_200_turn2', 'embedding_200_turn3'], axis=1)

        self.model = Sequential()

        features = Input(shape=(D.shape[1],), name="features_input")
        featuresL = Dense(512, activation='relu')(features)
        featuresL = Dense(512, activation='relu')(featuresL)
        featuresL = Dense(256, activation='relu')(featuresL)
        featuresL = Dense(256, activation='relu')(featuresL)
        featuresL = Dense(128, activation='relu')(featuresL)
        featuresL = Dense(128, activation='relu')(featuresL)
        # featuresL = Dense(64, activation='relu')(featuresL)
        # featuresL = Dense(64, activation='relu')(featuresL)
        # featuresL = Dense(32, activation='relu')(featuresL)
        # featuresL = Dense(32, activation='relu')(featuresL)
        
        input_turn1, lstm1_out = self.add_turn_layer("1")
        input_turn2, lstm2_out = self.add_turn_layer("2")
        input_turn3, lstm3_out = self.add_turn_layer("3")

        x = featuresL

        #x = concatenate([featuresL, lstm1_out, lstm2_out, lstm3_out])
        #x = Dense(128, activation='relu')(x)
        #x = Dense(128, activation='relu')(x)
        #x = Dense(128, activation='relu')(x)
        #x = Dense(128, activation='relu')(x)
        #x = Dense(128, activation='relu')(x)

        main_out = Dense(self.labels.shape[1], activation='softmax', name='output')(x)
        self.model = Model(inputs=[features], outputs=[main_out])
        # self.model.add(SpatialDropout1D(0.2))
        # self.model.add(Bidirectional(CuDNNLSTM(128, return_sequences=True)))
        # self.model.add(Bidirectional(CuDNNLSTM(64)))
        # self.model.add(Dropout(0.25))
        # self.model.add(Dense(units=5, activation='softmax'))
        
        # filepath="trained_models/"+self._name+"weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
        loaded = False
        filepath="trained_models/" + self._name + ".model"
        if os.path.isfile(filepath):
        #    loaded = True
            self.model.load_weights(filepath)
            print("Loaded model")
        #else:
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        

        adadelta = Adadelta(lr=1.0, rho=0.995, epsilon=None, decay=0.0)

        self.model.compile(loss='categorical_crossentropy', optimizer=adadelta, metrics = ['accuracy'])
        print(self.model.summary())
        tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

        if not loaded:
            self.model.fit([D[trainIdx]],
                            self.labels[trainIdx],
                            epochs=200,
                            batch_size=64,
                            #shuffle=True,
                            validation_data=([D[validationIdx]],self.labels[validationIdx]),
                            callbacks=[EarlyStopping(monitor='val_loss',patience=10),
                                checkpoint,
                                tensorboard
                            ])
            print("Done training")

    def test(self,D):
        #TODO
        pass
