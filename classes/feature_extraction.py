from abc import ABC, abstractmethod

class feature_extraction(ABC):

    def __init__(self,importance=0,name="base"):
        self._importance = importance
        self._name = name
        print(name)

    @staticmethod
    @abstractmethod
    def run(self, model, data, labels):
        pass