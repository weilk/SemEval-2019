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
        D = D.drop(output_emocontext,axis=1) # data 
        self.data = D
        
        self.model = Sequential()

        features = Input(shape=(D.shape[1],), name="features_input")
        featuresL = Dense(128, activation='relu')(features)
        featuresL = Dense(64, activation='relu')(featuresL)
        featuresL = Dense(32, activation='relu')(featuresL)
        
        
        
        main_out = Dense(self.labels.shape[1], activation='softmax', name='output')(featuresL)
        self.model = Model(inputs=[features], outputs=[main_out])
      
        loaded = False
        filepath="trained_models/" + self._name + ".model"
        if os.path.isfile(filepath):
        #    loaded = True
            self.model.load_weights(filepath)
            print("Loaded model")
        #else:
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        

        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
        adadelta = Adadelta(lr=1.0, rho=0.975, epsilon=None, decay=0.0)
        
        self.model.compile( 
                            loss='categorical_crossentropy', 
                            optimizer=adadelta,
                            metrics = ['accuracy'],
                        )
        print(self.model.summary())
        tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

        total = self.labels.shape[0]
        happyCount = np.where(np.isclose(self.labels[:,0],1.0))[0].shape[0]
        angryCount = np.where(np.isclose(self.labels[:,1],1.0))[0].shape[0]
        sadCount = np.where(np.isclose(self.labels[:,2],1.0))[0].shape[0]
        otherCount = np.where(np.isclose(self.labels[:,3],1.0))[0].shape[0]

        if not loaded:
            try:
                self.model.fit([D[trainIdx]],
                                self.labels[trainIdx],
                                epochs=200,
                                batch_size=32,
                                #shuffle=True,
                                validation_data=([D[validationIdx]],self.labels[validationIdx]),
                                callbacks=[
                                    checkpoint,
                                    tensorboard
                                ],
                                class_weight={
                                    0: total / happyCount,
                                    1: total / angryCount,
                                    2: total / sadCount,
                                    3: total / otherCount
                                })

                
            except KeyboardInterrupt:
                try:
                    print("Training the validation data for extra juice")
                    self.model.fit([D[validationIdx]],
                                    self.labels[validationIdx],
                                    epochs=5,
                                    batch_size=64,
                                    #shuffle=True,
                                    validation_data=([D[validationIdx]],self.labels[validationIdx]),
                                    callbacks=[
                                        checkpoint,
                                        tensorboard
                                    ],
                                    class_weight={
                                        0: total / happyCount,
                                        1: total / angryCount,
                                        2: total / sadCount,
                                        3: total / otherCount
                                    })
                    
                except KeyboardInterrupt:
                    print("stopped early")
            self.model.load_weights(filepath)
            print("Done training")

    def test(self,D):
        #TODO
        pass
