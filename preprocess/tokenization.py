from classes import preprocess
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

class tokenization(preprocess):

    def run(self,D,columns):
    	for column in columns:
            D["tokenize_" + column] = D[column].apply(word_tokenize)
