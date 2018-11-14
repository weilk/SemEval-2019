from classes import feature_extraction
from nltk.tokenize import word_tokenize
import string

class number_of_punctuation_in_words(feature_extraction):
    def run(self,D,columns):
        def custom_function(prop, vowels = True):
            number_of_punctuation = 0
            for word in prop:
                for char in word:
                    if char in string.punctuation:
                        number_of_punctuation += 1
                        
            return number_of_punctuation
        
        for column in columns:
            tokenized_words = D[column].apply(word_tokenize)
            D['number_of_punctuation_in_words_{}'.format(column)] = tokenized_words.apply(custom_function)
            print(D['number_of_punctuation_in_words_{}'.format(column)])