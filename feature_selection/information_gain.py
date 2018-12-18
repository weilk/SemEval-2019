from classes import feature_selection
from utils import *
import sklearn 
import numpy as np
import os
import pickle

class information_gain(feature_selection):
    
    def run(self,D,columns=[]):
        if os.path.exists("processed_data/ig_selection.fs"):
            return D[pickle.load(open("processed_data/ig_selection.fs","rb"))]
    
        ltemp = []

        for el in output_emocontext:
            if el != "label":
                ltemp.append(el)
        veganGains = []
        totalGains = None
        subset = D.drop(output_emocontext, axis=1).drop(input_emocontext, axis=1).drop(columns,axis=1)
        
        for l in ltemp:
            res=sklearn.feature_selection.mutual_info_classif(subset, D[l].values.astype("int"), discrete_features='auto', n_neighbors=subset.shape[1], copy=True, random_state=None)
            ig=np.array(res)    
            if totalGains is None and l != "label_others":
                totalGains=ig
            else:
                #if l != "label_others":
                totalGains=totalGains+ig
            #good = subset.columns.values[ig>0.0]
            #veganGains.append((l,ig,good))

        epsilon = 0.0 #np.mean(totalGains)
        good = subset.columns.values[totalGains>epsilon]
        good = good.tolist()
        good.extend(output_emocontext)
        good.extend(input_emocontext)
        good.extend(columns)

        pickle.dump(good,open("processed_data/ig_selection.fs","wb"))
        return D[good]

