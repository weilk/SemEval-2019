from classes import feature_extraction
import string
from re import search

class number_of_elongated_words(feature_extraction):
    
    def run(self,D,columns,changes):
    
        if not changes and self.load() == True:
            print("Loaded " + self._name + " from disk")
            return self.saved_data

        for column in columns:
            self.saved_data['number_of_elongated_words_{}'.format(column)] = D[column].apply(lambda x: len([y for y in x.split() if search(r'([a-zA-Z])\1\1+', y)]) )

        self.save()

        return self.saved_data


