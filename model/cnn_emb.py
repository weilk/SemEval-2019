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

class cnn_emb(model):
    
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
        return input_turn, LSTM(64, activation='relu')(LSTM(64, activation='relu',return_sequences=True)(LSTM(64, activation='relu',return_sequences=True)(LSTM(64, activation='relu',return_sequences=True)(emb))))

    def train(self, D,trainIdx,validationIdx, embedding_matrix, embedding_dim=200):
        self.embedding_matrix = embedding_matrix
        self.embedding_dim = embedding_dim

        self.labels = D[output_emocontext].values
        print(self.labels[0])
        print(self.labels[1])
        print(np.shape(self.labels))
        D = D.drop(output_emocontext,axis=1) # data 
        self.data = D
        # emb1_input = np.array([np.array(x) for x in D['embedding_200_turn1'].values])
        # emb2_input = np.array([np.array(x) for x in D['embedding_200_turn2'].values])
        # emb3_input = np.array([np.array(x) for x in D['embedding_200_turn3'].values])
        input_turn_length = len(self.data['embedding_200_turn1'][0]) + len(self.data['embedding_200_turn2'][0]) + len(self.data['embedding_200_turn3'][0])
        
        # print("Dropping emb cols: " )
        # D = D.drop(['embedding_200_turn1', 'embedding_200_turn2', 'embedding_200_turn3'], axis=1)

        self.model = Sequential()
        # self.model.add(Conv2D(32, (3,8), padding='same', activation="relu", input_shape=(3, input_turn_length, 1)))
        # self.model.add(Conv2D(32, (3,8), padding='same', activation="relu"))
        # self.model.add(Conv2D(32, (3,8), padding='same', activation="relu"))
        # self.model.add(MaxPooling2D(pool_size=(2,2), dim_ordering="th"))
        
        # self.model.add(Conv2D(64, (3,5), padding='same', activation="relu"))
        # self.model.add(Conv2D(64, (3,5), padding='same', activation="relu"))
        # self.model.add(Conv2D(64, (3,5), padding='same', activation="relu"))
        # self.model.add(MaxPooling2D(pool_size=(2,2), dim_ordering="th"))

        # self.model.add(Conv2D(128, (3,3), padding='same', activation="relu"))
        # self.model.add(Conv2D(128, (3,3), padding='same', activation="relu"))
        # self.model.add(Conv2D(128, (3,3), padding='same', activation="relu"))
        # self.model.add(MaxPooling2D(pool_size=(2,2), dim_ordering="th"))
        # self.model.add(Flatten())
        # self.model.add(Dense(128, activation='relu'))
        # self.model.add(Dense(1, activation='softmax'))
        
        self.model.add(Conv1D(32, 8, padding='same', activation="relu",
                                input_shape=(input_turn_length,1)))
        self.model.add(Conv1D(32, 8, padding='same', activation="relu"))
        self.model.add(Conv1D(32, 8, padding='same', activation="relu"))
        self.model.add(MaxPooling1D(pool_size=4))
        
        self.model.add(Conv1D(64, 8, padding='same', activation="relu"))
        self.model.add(Conv1D(64, 8, padding='same', activation="relu"))
        self.model.add(Conv1D(64, 8, padding='same', activation="relu"))
        self.model.add(MaxPooling1D(pool_size=4))

        self.model.add(Conv1D(128, 4, padding='same', activation="relu"))
        self.model.add(Conv1D(128, 4, padding='same', activation="relu"))
        self.model.add(Conv1D(128, 4, padding='same', activation="relu"))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(1, activation='softmax'))

        # features = Input(shape=(D.shape[1],), name="features_input")
        # featuresL = Dense(64, activation='relu')(features)
        # featuresL = Dense(64, activation='relu')(featuresL)
        # featuresL = Dense(64, activation='relu')(featuresL)
        # featuresL = Dense(64, activation='relu')(featuresL)
        # featuresL = Dense(64, activation='relu')(featuresL)
        # featuresL = Dense(64, activation='relu')(featuresL)
        # featuresL = Dense(64, activation='relu')(featuresL)
        # featuresL = Dense(64, activation='relu')(featuresL)
        
        # input_turn1, lstm1_out = self.add_turn_layer("1")
        # input_turn2, lstm2_out = self.add_turn_layer("2")
        # input_turn3, lstm3_out = self.add_turn_layer("3")

        # x = concatenate([featuresL, lstm1_out, lstm2_out, lstm3_out])
        # x = Dense(128, activation='relu')(x)
        # x = Dense(128, activation='relu')(x)
        # x = Dense(128, activation='relu')(x)
        # x = Dense(128, activation='relu')(x)
        # x = Dense(128, activation='relu')(x)

        # main_out = Dense(self.labels.shape[1], activation='softmax', name='output')(x)
        # self.model = Model(inputs=[features, input_turn1, input_turn2, input_turn3], outputs=[main_out])
        # self.model = Model(inputs=D, outputs=[x])
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
            # self.model.fit([D[trainIdx], emb1_input[trainIdx], emb2_input[trainIdx], emb3_input[trainIdx]],
            # print(np.shape(np.asarray([D[trainIdx]])))
            # print(np.shape(D[trainIdx]))
            # print(np.shape(D))
            # for index, row in D[trainIdx].iterrows():
            #     print(index)
            #     print(row['embedding_200_turn1'])
            #     print(type(row['embedding_200_turn1']))

            #     if index == 5:
            #         return
            train_data = np.array([np.concatenate((row['embedding_200_turn1'],row['embedding_200_turn2'],row['embedding_200_turn3']))
                            for index, row in D[trainIdx].iterrows()])
            validation_data = np.array([np.concatenate((row['embedding_200_turn1'],row['embedding_200_turn2'],row['embedding_200_turn3']))
                            for index, row in D[validationIdx].iterrows()])
            train_data = np.expand_dims(train_data, axis=2)
            validation_data = np.expand_dims(validation_data, axis=2)
            print(np.shape(train_data))
            self.model.fit(train_data,
                            self.labels[trainIdx],
                            epochs=200,
                            batch_size=32,
                            shuffle=True,
                            # validation_data=([D[validationIdx], emb1_input[validationIdx], emb2_input[validationIdx], emb3_input[validationIdx]],self.labels[validationIdx]),
                            validation_data=[validation_data],
                            callbacks=[EarlyStopping(patience=3),
                                checkpoint,
                                tensorboard
                            ]
                            )
            print("Done training")

    def test(self,D):
        #TODO
        pass
