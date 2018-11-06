from abc import ABC, abstractmethod
import pickle
import os

class model(ABC):

    def __init__(self,name="base"):        
        self._name = name


    @abstractmethod
    def forward_pass(self,D):
        pass


    @abstractmethod
    def train(self,D):
        pass

    
    @abstractmethod
    def test(self,D):
        pass

    
    def save(self):
        dir_path = "TAIP/SemEval-2019/trained_models/"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        fd = open("TAIP/SemEval-2019/trained_models/" + self._name + ".model", 'wb')
        pickle.dump(self.__dict__, fd)
        fd.close() 
    
    
    def load(self):
        try:
            fd = open("TAIP/SemEval-2019/trained_models/" + self._name + ".model", 'rb')
            tmp_dict = pickle.load(fd)
            fd.close()
        except IOError:
            return False

        self.__dict__.clear()
        self.__dict__.update(tmp_dict)
        
        return True