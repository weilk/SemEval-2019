from abc import ABC, abstractmethod

class feature_extraction(ABC):

    def __init__(self,importance=0,name="base",id=0):
        self._importance = importance
        self._name = name
        self._id = id


    @staticmethod
    @abstractmethod
    def run(self, model, data, labels):
        pass