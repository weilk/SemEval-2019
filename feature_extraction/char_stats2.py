from classes import feature_extraction
from nltk.tokenize import word_tokenize
from collections import Counter
from functools import reduce


def reduce_sets(func, setList):
	cc = list(setList)
	if len(cc) == 0: 
		return []
	return reduce(func, cc)

	
class char_stats2(feature_extraction):

	def consec_letters_set(self, word, l=2): 
		return set([word[s:s+l] for s in range(len(list(word))-l+1)])

	def words_consecutive_letters_set(self, words):
		return sorted(list(reduce_sets((lambda x, y: x|y), map(self.consec_letters_set, words))))
	
	def run(self, D, columns = []):
		for column in columns:
			tokenized_words = D[column].apply(lambda x: word_tokenize(x))
			stats = tokenized_words.apply(lambda x: self.words_consecutive_letters_set(x))
			alfabet = 'abcdefghijklmnopqrstuvwxyz'
			for comb in [a+b for a in alfabet for b in alfabet]:
				D['char_stats2_{}_{}'.format(column[-1:], comb)] = stats.apply(lambda x: int(comb in x))
			