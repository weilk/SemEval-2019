from classes import preprocess
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

class eliminate_stop_words(preprocess):

    def run(self,D,columns):  	    	
    	stop_words = set(stopwords.words('english'))
    	for column in columns:
            tokenized_words = D[column].apply(word_tokenize)
            D[column] = tokenized_words.apply(lambda x: TreebankWordDetokenizer().detokenize([w for w in x if not w in stop_words]))
            