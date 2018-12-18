from classes import postprocess
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from utils import *
import numpy as np
import os 
import pickle

class normalize(postprocess):
    def run(self,D,columns):
        
        normalization_intervals = []
        if os.path.exists("processed_data/normalization_intervals"):
            fd = open("processed_data/normalization_intervals",'rb')
            normalization_intervals = pickle.load(fd)

        if columns == None:
            if len(normalization_intervals) == 0:
                columns = D.drop(['embedding_200_turn1', 'embedding_200_turn2', 'embedding_200_turn3','id', 'label', 'turn1','turn2','turn3'],axis=1).columns
            else: 
                columns = D.drop(['embedding_200_turn1', 'embedding_200_turn2', 'embedding_200_turn3','id','turn1','turn2','turn3'],axis=1).columns

        
        if len(normalization_intervals) == 0:
            for column in columns:
                max_value = np.max(D[column].values)
                min_value = np.min(D[column].values)
                D[column] = (D[column].values - min_value) / (max_value - min_value + np.finfo(np.float32).eps)

                #D[column] = D[column].subtract(D[column].min())
                #D[column] = D[column].divide(D[column].max() - D[column].min())
                normalization_intervals.append((min_value, max_value))
        else:
            for i, column in enumerate(columns):
                max_value = normalization_intervals[i][1]
                min_value = normalization_intervals[i][0]
                D[column] = (D[column].values - min_value) / (max_value - min_value + np.finfo(np.float32).eps)

                #D[column] = D[column].subtract(D.intervals[i][0])
                #D[column] = D[column].divide(normalization_intervals[i][1] - normalization_intervals[i][0])
        
        fd = open("processed_data/normalization_intervals",'wb')
        pickle.dump(normalization_intervals,fd)
        fd.close()
        