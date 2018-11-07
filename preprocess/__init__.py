from preprocess.make_lower_case import *
from preprocess.eliminate_stop_words import *
from preprocess.replace_negation_words import *
from preprocess.tokenization import *

make_lower_case = make_lower_case(0, "make_lower_case")
eliminate_stop_words = eliminate_stop_words(-5, "eliminate_stop_words")
replace_negation_words = replace_negation_words(5, "replace_negation_words")
tokenization = tokenization(0, "tokenization")