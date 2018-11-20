from classes import preprocess
from nltk.tokenize import TweetTokenizer
import numpy as np
import pandas as pd

class embed_200(preprocess):

    def __init__(self, imp, name):
        preprocess.__init__(self, imp, name)
        self.load_embeddings()

    def load_embeddings(self):
        self.embeddings = dict()
        with open('embeddings/glove.twitter.27B.200d.txt', encoding="utf8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                self.embeddings[word] = coefs

    def run(self,D,columns):
        tokenizer = TweetTokenizer()
        for column in columns:
            embeddings = []
            for index in range(len(D[column])):
                words = tokenizer.tokenize(D[column][index])
                embedding = np.array(0)
                for word in words:
                    embedding = np.append(embedding, self.embeddings.get(word, np.zeros(200)))
                embeddings.append(embedding)
            D['embedding_200_{}'.format(column)] = pd.Series(embeddings)
        print(D.columns.values)
        print("embedded")
        return D