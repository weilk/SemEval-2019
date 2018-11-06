from classes import feature_extraction
import string

class number_of_words(feature_extraction):
    
    def run(self,D,columns = []):
        for column in columns:
            D['number_of_words_{}'.format(column)] = D[column].apply(lambda x:sum([i.strip(string.punctuation).isalpha() for i in x.split()]))