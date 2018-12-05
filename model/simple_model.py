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

class simple_model(model):
    
    def forward_pass(self,D):
              
        D = D.drop(['turn1', 'turn2', 'turn3'], axis=1)
        return self.model.predict([D])
    
    
    def train(self, D,trainIdx,validationIdx):
        print(D[output_emocontext].columns)
        self.labels = D[output_emocontext].values
        print(output_emocontext)
        D = D.drop(output_emocontext,axis=1) # data 
        self.data = D
        
        self.model = Sequential()

        features = Input(shape=(D.shape[1],), name="features_input")
        featuresL = Dense(512, activation='relu')(features)
        featuresL = Dropout(0.3)(featuresL)
        featuresL = Dense(512, activation='relu')(featuresL)
        featuresL = Dropout(0.3)(featuresL)
        featuresL = Dense(256, activation='relu')(featuresL)
        featuresL = Dropout(0.3)(featuresL)
        featuresL = Dense(256, activation='relu')(featuresL)
        featuresL = Dropout(0.3)(featuresL)
        featuresL = Dense(128, activation='relu')(featuresL)
        featuresL = Dropout(0.3)(featuresL)
        featuresL = Dense(128, activation='relu')(featuresL)
        
        x = featuresL

      
        main_out = Dense(self.labels.shape[1], activation='softmax', name='output')(x)
        self.model = Model(inputs=[features], outputs=[main_out])
      
        loaded = False
        filepath="trained_models/" + self._name + ".model"
        if os.path.isfile(filepath):
        #    loaded = True
            self.model.load_weights(filepath)
            print("Loaded model")
        #else:
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        

        adadelta = Adadelta(lr=1.0, rho=0.98, epsilon=None, decay=0.0)

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
                            callbacks=[EarlyStopping(monitor='val_loss',patience=2),
                                checkpoint,
                                tensorboard
                            ])
            print("Done training")

    def test(self,D):
        #TODO
        pass
