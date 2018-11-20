from classes import preprocess                                    
from autocorrect import spell
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import pandas as pd

emo = [':)', ':-)', ':))', ':-))', ';)', ';-)', ':P', ':p', ':-p', ':-P', ';^)', 'B-)', ':o)', ':(', ':-(']
 
class spellingcheck(preprocess):

    def run(self,D,columns):
        for column in columns:
            count = 0
            tokenized_words = D[column].apply(word_tokenize)
            all_props = []
            for llist in tokenized_words:
               temp = []
               for el in llist:       
                   if el not in emo:           
                       temp.append(spell(el))
                   else:
                       temp.append(el)
               all_props.append(TreebankWordDetokenizer().detokenize(temp))
               count = count + 1
            D[column] = pd.Series(all_props) 
            print(D[column])
               



