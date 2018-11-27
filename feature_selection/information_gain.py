from classes import feature_selection
from utils import *
import sklearn 

class information_gain(feature_selection):
    
    def run(self,D):
        ltemp = []
        for el in output_emocontext:
            if el != "label":
                ltemp.append(el)
        print(D.drop(output_emocontext, axis=1))
        print(D[ltemp])
        return sklearn.feature_selection.mutual_info_classif(D.drop(output_emocontext, axis=1).drop(input_emocontext, axis=1), D[ltemp], discrete_features='auto', n_neighbors=3, copy=True, random_state=None)