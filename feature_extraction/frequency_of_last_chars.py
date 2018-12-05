from classes import feature_extraction
from nltk.tokenize import word_tokenize
import json

alfabet = 'abcdefghijklmnopqrstuvwxyz'

class frequency_of_last_chars(feature_extraction):

    def custom_function(self,entry):
        frequency_vector = dict()
        for prop in entry:
            tokenized_words = word_tokenize(prop)
            for word in tokenized_words:
                only_ascii = [x for x in word[-3:] if ((ord(x) | 0x20) >= ord('a') and (ord(x) | 0x20) <= ord('z'))]
                if len(only_ascii) < 3:
                    continue
                for char in word[-3:]:
                    frequency_vector[char] = frequency_vector.setdefault(char, 0) + 1
        return frequency_vector
    
    def run(self,D,columns,changes):
    
        if not changes and self.load() == True:
            print("Loaded " + self._name + " from disk")
            return self.saved_data


        for i in range(26):
            self.saved_data['freq_of_last_chr_'+ chr(ord('a') + i)] = [0 for _ in range(len(D[columns[0]]) - 1)]

        for index_row in range(len(D[columns[0]])):
            list = []
            for index_column in range(len(columns)):
                list += [D[columns[index_column]][index_row]]

            freq = self.custom_function(list)
            for ch in freq:
                self.saved_data.loc['freq_of_last_chr_'+ch,index_row] = freq[ch]


        self.save()
        
        return self.saved_data
            
        
