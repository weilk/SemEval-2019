from classes import postprocess
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

class normalize(postprocess):
    def run(self,D,columns):
        if columns == None:
            columns = D.drop(['id', 'label', 'turn1','turn2','turn3'],axis=1).columns
        if not hasattr(D,'intervals'):
            D.intervals = []
            for column in columns:
                #D[column].subtract(D[column].min())
                #D[column].divide(D[column].max() - D[column].min())
                D[column] = D[column].subtract(D[column].min())
                D[column] = D[column].divide(D[column].max() - D[column].min())
                D.intervals.add(D[column].min(), D[column].max())
        else:
            for i, column in enumerate(columns):
                #D[column].subtract(D.intervals[i][0])
                #D[column].divide(D.intervals[i][1] - D.intervals[i][0])
                D[column] = D[column].subtract(D.intervals[i][0])
                D[column] = D[column].divide(D.intervals[i][1] - D.intervals[i][0])
            

        