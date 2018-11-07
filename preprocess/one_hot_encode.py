from classes import preprocess
import pandas as pd
from utils import *

class one_hot_encode(preprocess):

    def run(self,D,columns):
        encoded = pd.get_dummies(D[columns])
        D = pd.concat([D, encoded], axis=1)
        output_emocontext.extend(encoded.columns.values)
        return D