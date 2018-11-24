from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from classes import feature_selection
from keras.models import Sequential
import pandas as pd
import numpy as np
import keras
from utils import *

from model import simple_MLP
from keras.layers import Dense, Activation

class recursive_feature_elimination(feature_selection):
    
    def run(self, model, data, labels):
        global simple_MLP
        data = pd.DataFrame(data)
        labels = pd.DataFrame(labels)
        msk = np.random.rand(len(data)) < 0.8
        test_data = data[~msk]
        test_labels = labels[~msk]

        data = data[msk]
        labels = labels[msk]

        best_acc = model.test_diff(test_data, test_labels)[1]
        to_elim = "notnull"
        tempData = data.copy()
        
        while to_elim:
          to_elim = []
          for i in range(np.shape(tempData)[1]):
            data_feature_elim = tempData.drop(tempData.columns.values[i], axis=1)

            test_data_feature_elim = test_data.drop(test_data.columns.values[i], axis=1)

            copy_model = simple_MLP("simple_MLP")
            copy_model.train({"data": data_feature_elim, "labels": labels})

            current_acc = copy_model.test_diff(test_data_feature_elim, test_labels)[1]
            print("After eliminating feature: %s, acc is %s, best so far being %s"
                  % (tempData.columns.values[i], str(current_acc), str(best_acc)))
            if current_acc > best_acc:
              print("improved acc to %s" % str(current_acc))
              best_acc = current_acc
              to_elim.append(tempData.columns.values[i])
          tempData = tempData.drop(to_elim, axis=1)
          test_data = test_data.drop(to_elim, axis=1)
        return tempData.columns.values

    # def run(self,model,data,labels):
    #     print("feature extraction")
    #     model_copy_1 = keras.models.clone_model(model)
    #     model_copy_1.set_weights(model.get_weights())
    #     model_copy_1.add(Lambda(1, activation=np.argmax))
    #     # model_copy_1.add(Dense(1, activation=np.argmax))
    #     # model_2 = 
    #     # model_3 = 
    #     # model_4 = 
    #     rfe = RFE(model_copy_1, 4)
    #     # rfe =  RFECV(model, step=1, cv=5)
    #     print(np.shape(data))
    #     print(np.shape(labels))
    #     print(labels)
    #     rfe = rfe.fit(data, labels)
    #     # summarize the selection of the attributes
    #     print(rfe.support_)
    #     print(rfe.ranking_)
    #     print(dir(rfe))
    #     # return random.choice(D["x"])
    #     return sorted(zip(map(lambda x: round(x, 4), rlasso.scores_), 
    #              names), reverse=True)