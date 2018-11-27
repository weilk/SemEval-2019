from classes import feature_selection
from utils import *
import sklearn 
import numpy as np

class information_gain(feature_selection):
    
    def run(self,D):
        ltemp = []

        print(output_emocontext)
        for el in output_emocontext:
            if el != "label":
                ltemp.append(el)
        veganGains = []
        totalGains = None
        for l in ltemp:
            print(l)
            ig=np.array(sklearn.feature_selection.mutual_info_classif(D.drop(output_emocontext, axis=1).drop(input_emocontext, axis=1), D[l], discrete_features='auto', n_neighbors=3, copy=True, random_state=None))    
            if totalGains is None:
                totalGains=ig
            else:
                if l != "label_others":
                    totalGains=totalGains+ig
            good = D.drop(output_emocontext, axis=1).drop(input_emocontext, axis=1).columns.values[ig>0.0]
            print(len(D.drop(output_emocontext, axis=1).drop(input_emocontext, axis=1).columns.values))
            print(len(good))
            veganGains.append((l,ig,good))

        good = D.drop(output_emocontext, axis=1).drop(input_emocontext, axis=1).columns.values[totalGains>0.0]
        print(len(D.drop(output_emocontext, axis=1).drop(input_emocontext, axis=1).columns.values))
        print(len(good))
        
        return veganGains

