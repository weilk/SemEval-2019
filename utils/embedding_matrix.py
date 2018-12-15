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
    CACHE_FILE = "embedding_matrix.txt"
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

    def load_matrix(self, cache_file=CACHE_FILE):
        if path.isfile(cache_file):
            with open(cache_file, 'rb') as cf:
                matrix = pickle.load(cf)
                print("Loaded matrix")
                return np.array(matrix)

    def build_matrix(self, D, columns, save=False, load=False, embedding_dim=200, cache_file=CACHE_FILE):
        if load:
            return load_matrix(cache_file)
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
        with open("words_not_found_after_spell_check", "w", encoding="utf8") as f:
            print("words_not_found_after_spell_check")
            for word, i in word_index.items():
                if word not in self.embeddings:
                    f.write(word + "\n")
                embedding_matrix[i] = self.embeddings.get(word, np.random.randn(EMBEDDING_DIM))
        if save:
            self.save(embedding_matrix, cache_file)
        return np.array(embedding_matrix)

    def save(self, matrix, filename=CACHE_FILE):
        with open(filename, "wb") as f:
            pickle.dump(matrix, f)
            print("Dumped matrix")

