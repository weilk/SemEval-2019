from classes import feature_extraction
import string
from re import search

class number_of_elongated_words(feature_extraction):
    
    def run(self,D,columns = []):
        for column in columns:
            D['number_of_elongated_words_{}'.format(column)] = D[column].apply(lambda x: len([y for y in x.split() if search(r'([a-zA-Z])\1\1+', y)]) )


