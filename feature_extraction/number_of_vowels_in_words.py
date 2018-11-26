from classes import feature_extraction
from nltk.tokenize import word_tokenize

class number_of_vowels_in_words(feature_extraction):
    def run(self,D,columns):
        def custom_function_vowels(prop, vowels = True):
            total_size = 0
            number_of_vowels = 0
            for word in prop:
                for char in word:
                    if char.lower() in ['a','e','i','o','u']:
                        number_of_vowels += 1
                total_size += len(word)
                    
            if number_of_vowels == 0:
                return 0
            return int((number_of_vowels * 1.0 / total_size) * 100)
        
        for column in columns:
            tokenized_words = D[column].apply(word_tokenize)
            D['number_of_vowels_in_words_{}'.format(column)] = tokenized_words.apply(custom_function_vowels)