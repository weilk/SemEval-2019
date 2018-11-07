from feature_extraction.number_of_words import *
from feature_extraction.number_of_capitalized_words import *
from feature_extraction.number_negation_words import *
from feature_extraction.number_of_elongated_words import *
from feature_extraction.number_boosting_words import *
from feature_extraction.number_question_marks import *

number_of_words = number_of_words(0,"number_of_words")
number_of_capitalized_words = number_of_capitalized_words(0, "number_of_capitalized_words")
number_of_elongated_words = number_of_elongated_words(0, "number_of_elongated_words")
number_negation_words = number_negation_words(0, "number_negation_words")
number_boosting_words = number_boosting_words(0, "number_boosting_words")
number_question_marks = number_question_marks(0, "number_question_marks")
