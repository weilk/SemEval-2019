from classes import feature_extraction
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

happy_emo_list = [':)', ':-)', ':))', ':-))', ';)', ';-)', ':P', ':p', ':-p', ':-P', ';^)', 'B-)', ':o)']

class number_happy_emoticons(feature_extraction):

    def run(self,D,columns):
        for column in columns:
            tokenized_words = D[column].apply(word_tokenize)
            D['number_of_happy_emoticons_{}'.format(column)] = tokenized_words.apply(lambda x: (len([w for w in x if w in happy_emo_list]) > 0) )
        
        

