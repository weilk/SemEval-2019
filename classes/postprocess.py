from abc import ABC, abstractmethod
from classes.savable import savable
import pandas as pd

class postprocess(ABC, savable):
    
    def __init__(self,importance=0,name="base",id=0):
        self._importance = importance
        self._name = name
        self._id = id
        self.saved_data = pd.DataFrame()   

    @staticmethod
    @abstractmethod
    def run(self,D,columns):
        pass