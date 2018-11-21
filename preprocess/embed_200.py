from classes import preprocess
from nltk.tokenize import TweetTokenizer
import numpy as np
import pandas as pd
from keras.preprocessing import sequence 
import codecs

MAXLEN_TURN1 = 65
MAXLEN_TURN2 = 21
MAXLEN_TURN3 = 143

EMBEDDING_DIM = 200

class embed_200(preprocess):

    def __init__(self, imp, name):
        preprocess.__init__(self, imp, name)
        self.load_embeddings()
        print("Loaded %s embeddings" % str(len(self.embeddings.keys())))

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
            emb.append(np.zeros(EMBEDDING_DIM))
        return emb

    def run(self,D,columns):
        # with codecs.open("words_not_found", "w", "utf-8") as nf: 
            tokenizer = TweetTokenizer()
            for column in columns:
                embeddings = []
                for index in range(len(D[column])):
                    words = tokenizer.tokenize(D[column][index])
                    phrase_embedding = []
                    for word in words:
                        phrase_embedding.append(self.embeddings.get(word, np.zeros(200)))
                        if self.embeddings.get(word) is None:
                            nf.write(word + "\n")

                    
                    if column == 'turn1':
                        phrase_embedding = self.pad_phrase_embedding(phrase_embedding, MAXLEN_TURN1)
                        # assert np.shape(phrase_embedding) == (MAXLEN_TURN1, EMBEDDING_DIM)

                    if column == 'turn2':
                        phrase_embedding = self.pad_phrase_embedding(phrase_embedding, MAXLEN_TURN2)

                    if column == 'turn3':
                        phrase_embedding = self.pad_phrase_embedding(phrase_embedding, MAXLEN_TURN3)
                    
                    embeddings.append(phrase_embedding)
                print(column)
                with open("embedd_dump"+column, "w") as f:
                    for i, embedding in enumerate(embeddings):
                        f.write("%s\n" % str(i))
                        f.write(" ".join(str(x) for x in embedding))

                print(np.shape(np.array(embeddings)))
                D['embedding_200_{}'.format(column)] = pd.Series(embeddings)
                print(np.shape(pd.Series(embeddings)))

        print(D.columns.values)
        print("embedded")
        print("shapes of embeddings:")
        print(D['embedding_200_turn1'].shape)
        print(D['embedding_200_turn2'].shape)
        print(D['embedding_200_turn3'].shape)

        import ipdb
        ipdb.set_trace(context=10)
        return D