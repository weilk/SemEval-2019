from classes import preprocess
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

class number_exclamation_marks(preprocess):

    def run(self,D,columns):
        for column in columns:
            tokenized_words = D[column].apply(word_tokenize)
            D['number_of_question_marks_{}'.format(column)] = tokenized_words.apply(lambda x: len([w for w in x if w == "!"]))
        
        

