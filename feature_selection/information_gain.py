from classes import feature_selection
import sklearn 

class information_gain(feature_selection):
    
    def run(self,D):
        return sklearn.feature_selection.mutual_info_classif(D.drop(output_emocontext), D[output_emocontext], discrete_features='auto', n_neighbors=3, copy=True, random_state=None)