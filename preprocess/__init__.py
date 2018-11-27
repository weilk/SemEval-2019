from preprocess.make_lower_case import *
from preprocess.eliminate_stop_words import *
from preprocess.replace_negation_words import *
from preprocess.tokenization import *
from preprocess.one_hot_encode import *
from preprocess.spellingcheck import *
from preprocess.extract_redundant_words import *

make_lower_case = make_lower_case(0, "make_lower_case")
eliminate_stop_words = eliminate_stop_words(-5, "eliminate_stop_words")
replace_negation_words = replace_negation_words(5, "replace_negation_words")
tokenization = tokenization(0, "tokenization")
one_hot_encode = one_hot_encode(-100, "one_hot_encode")
spellingcheck = spellingcheck(50, "spellingcheck")
extract_redundant_words = extract_redundant_words(-5, "extract_redundant_words")
