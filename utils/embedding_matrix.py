import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import codecs
from utils import max_no_words
import pickle
from os import path

MAXLEN_TURN1 = 65
MAXLEN_TURN2 = 21
MAXLEN_TURN3 = 143

EMBEDDING_DIM = 200

class embedding_matrix():

    def load_embeddings(self):
        self.embeddings = dict()
        with open('embeddings/glove.twitter.27B.200d.txt', encoding="utf8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                self.embeddings[word] = coefs

    def pad_phrase_embedding(self, emb, dim):
        n = len(emb)
        for i in range(dim-n):
            emb = np.append(emb, np.zeros(EMBEDDING_DIM))
        return emb

    def build_matrix(self, D, columns, embedding_dim=200, cache_file="matrix"):
        if path.isfile(cache_file):
            with open(cache_file, 'rb') as cf:
                matrix = pickle.load(cf)
                print("Loaded matrix")
                return np.array(matrix)
        self.load_embeddings()
        print("Loaded %s embeddings" % str(len(self.embeddings.keys())))

        tokenizer = Tokenizer()
        self.max_words_per_turn = max_no_words.get_no_words()
        for column in columns:
            tokenizer.fit_on_texts(D[column])

        word_index = tokenizer.word_index

        num_words = len(word_index)
        print('Found %s unique tokens.' % num_words)

        embedding_matrix = np.zeros((num_words + 1, EMBEDDING_DIM))
        
        for word, i in word_index.items():
            embedding_matrix[i] = self.embeddings.get(word, np.random.randn(EMBEDDING_DIM))

        with open(cache_file, 'wb') as cf:
            pickle.dump(embedding_matrix, cf)
            print("Dumped matrix")
        return np.array(embedding_matrix)
