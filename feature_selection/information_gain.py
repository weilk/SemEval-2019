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
            #print(l)
            first_drop = D.drop(output_emocontext, axis=1)
            sec_drop = first_drop.drop(input_emocontext, axis=1)
            print(sec_drop.shape)
            print(D[l].shape)
            res=sklearn.feature_selection.mutual_info_classif(sec_drop.values, D[l].values, discrete_features='auto', n_neighbors=3, copy=True, random_state=None)
            ig=np.array(res)    
            if totalGains is None:
                totalGains=ig
            else:
                if l != "label_others":
                    totalGains=totalGains+ig
            good = D.drop(output_emocontext, axis=1).drop(input_emocontext, axis=1).columns.values[ig>0.0]
            #print(len(D.drop(output_emocontext, axis=1).drop(input_emocontext, axis=1).columns.values))
            #print(len(good))
            veganGains.append((l,ig,good))

        good = D.drop(output_emocontext, axis=1).drop(input_emocontext, axis=1).columns.values[totalGains>0.0]
        #print(len(D.drop(output_emocontext, axis=1).drop(input_emocontext, axis=1).columns.values))
        #print(len(good))
        
        #return veganGains
        return D[veganGains[1][2]]

