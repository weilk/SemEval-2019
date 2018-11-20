from classes import feature_extraction
from nltk.tokenize import word_tokenize

class number_boosting_words(feature_extraction):

    def run(self,D,columns):
        for column in columns:
            tokenized_words = D[column].apply(word_tokenize)
            D['number_of_boosting_words_{}'.format(column)] = tokenized_words.apply(lambda x: len([w for w in x if w in list_boosting]))
        

