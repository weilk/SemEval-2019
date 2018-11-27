from classes import feature_extraction
from nltk.tokenize import word_tokenize

class number_of_consonants_in_words(feature_extraction):

    def run(self,D,columns,changes):
    
        if not changes and self.load() == True:
            print("Loaded " + self._name + " from disk")
            return self.saved_data
                   
        def custom_function_consonants(prop):
            total_size = 0
            number_of_consonants = 0
            for word in prop:
                for char in word:
                    if char.lower() not in ['a','e','i','o','u'] and char.lower() >= 'a' and char.lower() <= 'z':
                        number_of_consonants += 1
                total_size += len(word)
                    
            if number_of_consonants == 0:
                return 0
            return int((number_of_consonants * 1.0 / total_size) * 100)
        
        for column in columns:
            tokenized_words = D[column].apply(word_tokenize)
            self.saved_data['number_of_consonants_in_words_{}'.format(column)] = tokenized_words.apply(custom_function_consonants)


        self.save()
        
        return self.saved_data 
