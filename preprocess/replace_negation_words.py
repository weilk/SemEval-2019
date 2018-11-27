from classes import preprocess
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

negations = ['no', 'none', 'no one', 'nobody', 'nothing', 'neither', 'nowhere', 'never', 'should not', 'shouldnt', 'nor', 'would not', 'wouldnt', 'could not', 'couldnt', 'must not', 'mustnt']

class replace_negation_words(preprocess):

    def run(self,D,columns):
    	for column in columns:
            tokenized_words = D[column].apply(word_tokenize)
            D[column] = tokenized_words.apply(lambda x: TreebankWordDetokenizer().detokenize(["not" if w.lower() in negations else w for w in x ]))
        