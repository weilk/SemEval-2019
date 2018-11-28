from classes import feature_extraction
import string
from feature_extraction import *

class percentage_capitalized(feature_extraction):

    def run(self,D,columns,changes):
    
        if not changes and self.load() == True:
            print("Loaded " + self._name + " from disk")
            return self.saved_data

        for column in columns:
            x = D[column].apply(lambda x:sum([i.strip(string.punctuation).isalpha() for i in x.split() if i.strip(string.punctuation).isalpha() and i.strip(string.punctuation).upper() == i.strip(string.punctuation) and len(i.strip(string.punctuation)) > 0]))
            y = D[column].apply(lambda x:sum([i.strip(string.punctuation).isalpha() for i in x.split()]))
            self.saved_data['percentage_of_capitalized_{}'.format(column)] = x/y

        self.save()
        
        return self.saved_data
                                                               



        

