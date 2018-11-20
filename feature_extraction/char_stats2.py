from classes import feature_extraction
from nltk.tokenize import word_tokenize
from collections import Counter
from functools import reduce

class char_stats2(feature_extraction):

	def consec_letters_set(self, word, l=2): 
		return set([word[s:s+l] for s in range(len(word)-l+1)])

	def words_consecutive_letters_set(self, words): 
		return sorted(list(reduce((lambda x, y: x|y), map(self.consec_letters_set, words))))
	
	def run(self, D, columns = []):
		for column in columns:
			tokenized_words = D[column].apply(word_tokenize)
			stats = tokenized_words.apply(self.words_consecutive_letters_set)
			alfabet = 'abcdefghijklmnopqrstuvwxyz'
			for comb in [a+b for a in alfabet for b in alfabet]:
				D['char_stats2_{}_{}'.format(column[-1:], comb)] = stats.apply(lambda x: int(comb in x))
			