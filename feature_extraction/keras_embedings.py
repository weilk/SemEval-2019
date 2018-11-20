from classes import feature_extraction
import string
import pandas as pd
from keras.preprocessing import sequence 
from keras.preprocessing.text import one_hot 
from re import search

class keras_embedings(feature_extraction):
    
    def run(self,D,columns = [], vocab_size = 300, max_review_length = 200, embedding_vector_length = 32 ):
        for column in columns:
            pass 
            #print(sequence.pad_sequences([one_hot(d, vocab_size) for d in D[column]], maxlen=max_review_length))


