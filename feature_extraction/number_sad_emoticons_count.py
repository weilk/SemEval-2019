from classes import feature_extraction
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

sad_emo_list = [':(', ':-(']

class number_sad_emoticons_count(feature_extraction):

    def run(self,D,columns):
        for column in columns:
            tokenized_words = D[column].apply(word_tokenize)
            D['number_of_sad_emoticons_count_{}'.format(column)] = tokenized_words.apply(lambda x: len([w for w in x if w in sad_emo_list]))
        
        

