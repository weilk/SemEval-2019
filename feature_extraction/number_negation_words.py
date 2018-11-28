from classes import feature_extraction
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

class number_negation_words(feature_extraction):

    def run(self,D,columns,changes):
    
        if not changes and self.load() == True:
            print("Loaded " + self._name + " from disk")
            return self.saved_data

        for column in columns:
            tokenized_words = D[column].apply(word_tokenize)
            self.saved_data['number_of_negated_words_{}'.format(column)] = tokenized_words.apply(lambda x: len([w for w in x if w == "not"]))

        self.save()

        return self.saved_data
        


