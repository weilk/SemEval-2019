from classes import feature_extraction
import string

class number_of_words(feature_extraction):
    
    def run(self,D,columns,changes):
    
        if not changes and self.load() == True:
            print("Loaded " + self._name + " from disk")
            return self.saved_data

        for column in columns:
            self.saved_data['number_of_words_{}'.format(column)] = D[column].apply(lambda x:sum([i.strip(string.punctuation).isalpha() for i in x.split()]))

        self.save()

        return self.saved_data