from classes import feature_extraction
from nltk.tokenize import word_tokenize

class number_of_vowels_in_words(feature_extraction):
    def run(self,D,columns):
        def custom_function_vowels(prop, vowels = True):
            number_of_vowels = 0
            number_of_consonants = 0
            for word in prop:
                for char in word:
                    if char.lower() in ['a','e','i','o','u']:
                        number_of_vowels += 1
                    elif char.lower() >= 'a' and char.lower() <= 'z':
                        number_of_consonants += 1
                    
            if vowels:
                if number_of_vowels== 0:
                    return 0
                return int((number_of_vowels * 1.0 / (number_of_vowels + number_of_consonants)) * 100)
                    
            else:
                if number_of_consonants == 0:
                    return 0
                return int((number_of_consonants * 1.0 / (number_of_vowels + number_of_consonants)) * 100)
                
        def custom_function_consonants(prop, vowels = False):
            number_of_vowels = 0
            number_of_consonants = 0
            for word in prop:
                for char in word:
                    if char.lower() in ['a','e','i','o','u']:
                        number_of_vowels += 1
                    elif char.lower() >= 'a' and char.lower() <= 'z':
                        number_of_consonants += 1
                    
            if vowels:
                if number_of_vowels== 0:
                    return 0
                return int((number_of_vowels * 1.0 / (number_of_vowels + number_of_consonants)) * 100)
                    
            else:
                if number_of_consonants == 0:
                    return 0
                return int((number_of_consonants * 1.0 / (number_of_vowels + number_of_consonants)) * 100)
       


        
        for column in columns:
            tokenized_words = D[column].apply(word_tokenize)
            D['number_of_consonants_in_words_{}'.format(column)] = tokenized_words.apply(custom_function_consonants)
            D['number_of_vowels_in_words_{}'.format(column)] = tokenized_words.apply(custom_function_vowels)