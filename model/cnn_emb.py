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

        self.data = np.array([np.concatenate((row['embedding_200_turn1'],row['embedding_200_turn2'],row['embedding_200_turn3']))
                            for index, row in D.iterrows()])
        happy = self.happy_model.predict(self.data)
        sad = self.sad_model.predict(self.data)
        angry = self.angry_model.predict(self.data)
        others = self.others_model.predict(self.data)
        predictions = np.concatenate((happy, sad, angry, others), axis=1)
        print(np.shape(predictions))
        print(predictions)
        return predictions
    

    def add_turn_layer(self, index):
        input_turn_length = len(self.data['embedding_200_turn%s' % index][0])
        input_turn = Input(shape=(input_turn_length,), name='turn%s_input' % index)
        emb = Embedding(input_dim=np.shape(self.embedding_matrix)[0],
                            output_dim=self.embedding_dim,
                            embeddings_initializer=Constant(self.embedding_matrix),
                            input_length=input_turn_length,
                            trainable=True)(input_turn)
        return input_turn, LSTM(64, activation='relu')(LSTM(64, activation='relu',return_sequences=True)(LSTM(64, activation='relu',return_sequences=True)(LSTM(64, activation='relu',return_sequences=True)(emb))))

    def _train_model(self, label,  D,trainIdx,validationIdx, embedding_matrix, embedding_dim=200, load=True):
        print("training %s" % label)
        print(D.columns)
        # print(D.drop(output_emocontext))
        self.data = np.array([np.concatenate((row['embedding_200_turn1'],row['embedding_200_turn2'],row['embedding_200_turn3']))
                            for index, row in D[trainIdx].iterrows()])
        self.val_data = np.array([np.concatenate((row['embedding_200_turn1'],row['embedding_200_turn2'],row['embedding_200_turn3']))
                    for index, row in D[validationIdx].iterrows()])
        self.labels = np.array(pd.DataFrame([0 if x[0]==label else 1 for x in D[output_emocontext].values])[trainIdx])
        self.labels = np.reshape(self.labels, (np.shape(self.labels)[0],))

        self.val_labels = np.array(pd.DataFrame([0 if x[0]==label else 1 for x in D[output_emocontext].values])[validationIdx])
        self.val_labels = np.reshape(self.val_labels, (np.shape(self.val_labels)[0],))

        input_turn_length = len(D['embedding_200_turn1'][0]) + len(D['embedding_200_turn2'][0]) + len(D['embedding_200_turn3'][0])
        num_filters = 10

        model = Sequential()
        model.add(Embedding(np.shape(embedding_matrix)[0], embedding_dim,
                  weights=[embedding_matrix], input_length=input_turn_length, trainable=True))
        model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
        model.add(MaxPooling1D(2))
        model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
        model.add(GlobalMaxPooling1D())
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.0004)))
        model.add(Dense(1, activation='sigmoid'))  #multi-label (k-hot encoding)

        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
        model.summary()
      
        filepath="trained_models/" + self._name + "-" + label + ".model"

        if load:
            model.load_weights(filepath)
            print("Loaded model")
            return model

        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        
        tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

        model.fit(self.data,
                self.labels,
                epochs=10,
                batch_size=32,
                shuffle=True,
                validation_data=(self.val_data, self.val_labels),
                callbacks=[EarlyStopping(patience=3),
                    checkpoint,
                    tensorboard
                ])
        return model

    def train(self, D,trainIdx,validationIdx, embedding_matrix, embedding_dim=200):
        self.happy_model = self._train_model("happy",D,trainIdx,validationIdx, embedding_matrix, embedding_dim)
        self.sad_model = self._train_model("sad",D,trainIdx,validationIdx, embedding_matrix, embedding_dim)
        self.angry_model = self._train_model("angry",D,trainIdx,validationIdx, embedding_matrix, embedding_dim)
        self.others_model = self._train_model("others",D,trainIdx,validationIdx, embedding_matrix, embedding_dim)

    def test(self,D):
        #TODO
        pass
