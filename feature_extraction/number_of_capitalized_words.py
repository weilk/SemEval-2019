from classes import feature_extraction
import string

class number_of_capitalized_words(feature_extraction):
    
    def run(self,D,columns = []):
        for column in columns:
            D['number_of_capitalized_words_{}'.format(column)] = D[column].apply(lambda x:sum([i.strip(string.punctuation).isalpha() for i in x.split() if i.strip(string.punctuation).isalpha() and i.strip(string.punctuation).upper() == i.strip(string.punctuation) and len(i.strip(string.punctuation)) == 0]))