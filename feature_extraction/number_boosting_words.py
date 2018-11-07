from classes import preprocess
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

list_boosting = ['almost', 'absolutely', 'awfully', 'badly', 'barely', 'completely', 'decidedly', 'deeply', 'enough', 'enormously', 'entirely', 'extremely',
'fairly', 'far', 'fully', 'greatly', 'hardly', 'highly', 'how', 'incredibly', 'indeed', 'intensely', 'just', 'least', 'less', 'little', 'lots', 'most', 'much', 'nearly',
'perfectly', 'positively', 'practically', 'pretty', 'purely', 'quite', 'rather', 'really', 'scarcely', 'simply', 'so', 'somewhat', 'strongly', 'terribly', 'thoroughly'
'too', 'totally', 'utterly', 'very', 'virtually', 'well']

class number_boosting_words(preprocess):

    def run(self,D,columns):
        for column in columns:
            tokenized_words = D[column].apply(word_tokenize)
            D['number_of_boosting_words_{}'.format(column)] = tokenized_words.apply(lambda x: len([w for w in x if w in list_boosting]))
        

