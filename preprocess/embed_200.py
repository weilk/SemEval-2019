from classes import preprocess
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from utils import max_no_words
import codecs

MAXLEN_TURN1 = 65
MAXLEN_TURN2 = 21
MAXLEN_TURN3 = 143

EMBEDDING_DIM = 200

class embed_200(preprocess):

    def run(self,D,columns):
        # self.load_embeddings()
        # print("Loaded %s embeddings" % str(len(self.embeddings.keys())))
        tokenizer = Tokenizer()
        self.max_words_per_turn = max_no_words.get_no_words()
        print("Max no words: %s" % str(self.max_words_per_turn))
        for column in columns:
            tokenizer.fit_on_texts(D[column])
            D['embedding_200_{}'.format(column)] = tokenizer.texts_to_sequences(D[column].values)
            D['embedding_200_{}'.format(column)] = list(pad_sequences(D['embedding_200_{}'.format(column)], self.max_words_per_turn[column]))
        return D
