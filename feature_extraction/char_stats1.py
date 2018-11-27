from classes import feature_extraction
from nltk.tokenize import word_tokenize
from collections import Counter

class char_stats1(feature_extraction):

    def only_last_3_letters(self, word):
        return ''.join([ch for ch in word.lower() if 'a'<=ch and ch<='z'])[-3:]

    def charstats1(self, words):
        words = list(filter(lambda w: len(w)==3, list(map(self.only_last_3_letters, words))))
        return dict(Counter(''.join(words)))
    
    def run(self,D,columns,changes):
    
        if not changes and self.load() == True:
            print("Loaded " + self._name + " from disk")
            return self.saved_data

        for column in columns:
            tokenized_words = D[column].apply(lambda x: word_tokenize(x))
            stats = tokenized_words.apply(lambda x: self.charstats1(x))
            for ordch in range(ord('a'), ord('z')+1):
                ch = chr(ordch)
                self.saved_data['char_stats1_{}_{}'.format(column[-1:], ch)] = stats.apply(lambda x: x.setdefault(ch, 0))
        
        self.save()

        return self.saved_data