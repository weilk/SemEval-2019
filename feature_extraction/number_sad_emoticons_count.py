from classes import feature_extraction
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

sad_emo_list = [':(', ':-(']

class number_sad_emoticons_count(feature_extraction):

    def run(self,D,columns,changes):
    
        if not changes and self.load() == True:
            print("Loaded " + self._name + " from disk")
            return self.saved_data

        for column in columns:
            tokenized_words = D[column].apply(word_tokenize)
            self.saved_data['number_of_sad_emoticons_count_{}'.format(column)] = tokenized_words.apply(lambda x: len([w for w in x if w in sad_emo_list]))
        
        self.save()

        return self.saved_data

