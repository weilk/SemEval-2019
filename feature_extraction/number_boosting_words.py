from classes import feature_extraction
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

class number_boosting_words(feature_extraction):

    def run(self,D,columns,changes):
    
        if not changes and self.load() == True:
            print("Loaded " + self._name + " from disk")
            return self.saved_data

        for column in columns:
            tokenized_words = D[column].apply(word_tokenize)
            self.saved_data['number_of_boosting_words_{}'.format(column)] = tokenized_words.apply(lambda x: len([w for w in x if w in list_boosting]))


        self.save()
        
        return self.saved_data
        

