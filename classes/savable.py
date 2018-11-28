import pickle
import os
import pandas as pd

class savable():

    def __init__(self, name="base"):        
        self._name = name

    def save(self):
        dir_path = "processed_data"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        k = ""
        if self.test:
            k = "t_"
        fd = open("processed_data/"+ k + self._name + ".data", 'wb')
        pickle.dump(self.__dict__, fd)
        fd.close() 
    
    
    def load(self):
        k = ""
        if self.test:
            k = "t_"
        try:
            fd = open("processed_data/"+ k + self._name + ".data", 'rb')
            tmp_dict = pickle.load(fd)
            fd.close()
        except IOError:
            return False

        self.__dict__.clear()
        self.__dict__.update(tmp_dict)
        
        return True