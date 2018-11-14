from classes import feature_extraction
from nltk.tokenize import word_tokenize

class number_of_capitals_in_words(feature_extraction):
    def run(self,D,columns):
        def custom_function(prop):
            number_of_wrong_spelled_words = 0
            for word in prop:
                if len(word) < 3:
                    continue
                    
                counter_of_big_letters = 0
                for x in word[1:]:
                    if ord(x) >= 65 and ord(x) <= 90:
                        counter_of_big_letters += 1
                
                if counter_of_big_letters > 0 and counter_of_big_letters <= len(word)-2:
                    number_of_wrong_spelled_words += 1
                    
            return number_of_wrong_spelled_words
                
        for column in columns:
            tokenized_words = D[column].apply(word_tokenize)
            D['number_of_capitals_in_words_{}'.format(column)] = tokenized_words.apply(custom_function)
