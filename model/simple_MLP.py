from classes import model
import pandas as pd
import numpy as np
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.preprocessing.text import one_hot
from utils import *

class simple_MLP(model):
    
    def forward_pass(self,D):
        return D 
    
    def train(self,D):
        self.data = D["data"]
        self.labels = D["labels"]

        self.model = Sequential()
        self.model.add(Dense(32, activation='relu', input_dim=self.data.shape[1]))
        self.model.add(Dense(len(output_emocontext), activation='softmax'))
        
        self.model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

        self.model.fit(self.data, self.labels, epochs=5, batch_size=32)
        print("Done training")

    def test(self,D):
        self.data = D.drop(output_emocontext,axis=1).values
        self.labels = pd.get_dummies(D[output_emocontext])
        results = self.model.evaluate(self.data, self.labels, batch_size=128)
        print("Done testing")
        return results

    def test_diff(self,data, labels):
        results = self.model.evaluate(data, labels, batch_size=128)
        print("Done testing")
        return results

    def save(self):
        self.model.save(self.model_file_name)
        print("Saved model to disk")

    def load(self):
        print("loading")
        self.model = load_model(self.model_file_name)
        print("Loaded model from disk")
