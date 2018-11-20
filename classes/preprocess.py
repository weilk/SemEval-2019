from abc import ABC, abstractmethod

class preprocess(ABC):
    
    def __init__(self,importance=0,name="base"):
        self._importance = importance
        self._name = name
        print(name)


    @staticmethod
    @abstractmethod
    def run(self,D,columns):
        pass