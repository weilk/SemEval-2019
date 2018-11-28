from classes import feature_selection
from utils import *
import sklearn 
import numpy as np

class information_gain(feature_selection):
    
    def run(self,D,columns=[]):
        ltemp = []

        for el in output_emocontext:
            if el != "label":
                ltemp.append(el)
        veganGains = []
        totalGains = None
        subset = D.drop(output_emocontext, axis=1).drop(input_emocontext, axis=1).drop(columns,axis=1)
        for l in ltemp:
            res=sklearn.feature_selection.mutual_info_classif(subset, D[l].values, discrete_features='auto', n_neighbors=3, copy=True, random_state=None)
            ig=np.array(res)    
            if totalGains is None and l != "label_others":
                totalGains=ig
            else:
                if l != "label_others":
                    totalGains=totalGains+ig
            #good = subset.columns.values[ig>0.0]
            #veganGains.append((l,ig,good))

        epsilon = np.median(totalGains)
        good = subset.columns.values[totalGains>epsilon]
        good = good.tolist()
        good.extend(output_emocontext)
        good.extend(input_emocontext)
        good.extend(columns)
        return D[good]

