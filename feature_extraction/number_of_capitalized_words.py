from classes import feature_extraction
import string

class number_of_capitalized_words(feature_extraction):
    
    def run(self,D,columns,changes):
    
        if not changes and self.load() == True:
            print("Loaded " + self._name + " from disk")
            return self.saved_data

        for column in columns:
            self.saved_data['number_of_capitalized_words_{}'.format(column)] = D[column].apply(lambda x:sum([i.strip(string.punctuation).isalpha() for i in x.split() if i.strip(string.punctuation).isalpha() and i.strip(string.punctuation).upper() == i.strip(string.punctuation) and len(i.strip(string.punctuation)) > 0]))

        self.save()
        
        return self.saved_data