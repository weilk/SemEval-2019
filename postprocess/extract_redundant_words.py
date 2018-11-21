from classes import postprocess
import nltk
import emoji
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer


emo = [':)', ':-)', ':))', ':-))', ';)', ';-)', ':P', ':p', ':-p', ':-P', ';^)', 'B-)', ':o)', ':(', ':-(']

class extract_redundant_words(postprocess):

    def run(self,D,columns):
    	for column in columns:
            tokenized_words = D[column].apply(word_tokenize)
            D[column] = tokenized_words.apply(lambda x: TreebankWordDetokenizer().detokenize([w for w in x if w not in emo and w not in emoji.UNICODE_EMOJI]))
        