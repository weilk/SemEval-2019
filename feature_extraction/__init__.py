from feature_extraction.number_of_words import *
from feature_extraction.number_of_capitalized_words import *
from feature_extraction.number_of_elongated_words import *
from feature_extraction.number_exclamation_marks import *
from feature_extraction.number_boosting_words import *
from feature_extraction.number_question_marks import *
from feature_extraction.number_negation_words import *
from feature_extraction.percentage_capitalized import *
from feature_extraction.frequency_of_last_chars import *
from feature_extraction.number_of_vowels_in_words import *
from feature_extraction.number_of_consonants_in_words import *
from feature_extraction.number_of_punctuation_in_words import *
from feature_extraction.keras_embedings import *
from feature_extraction.number_happy_emoticons import *
from feature_extraction.number_sad_emoticons import *
from feature_extraction.number_happy_emoticons_count import *
from feature_extraction.number_sad_emoticons_count import *
from feature_extraction.bad_words import *
from feature_extraction.number_of_capitals_in_words import *
from feature_extraction.char_stats1 import *
from feature_extraction.char_stats2 import *


number_of_words = number_of_words(5,"number_of_words",1)
number_of_capitalized_words = number_of_capitalized_words(5, "number_of_capitalized_words",2)
number_of_elongated_words = number_of_elongated_words(0, "number_of_elongated_words",3)
number_negation_words = number_negation_words(0, "number_negation_words",4)
number_boosting_words = number_boosting_words(0, "number_boosting_words",5)
number_exclamation_marks = number_exclamation_marks(0, "number_exclamation_marks",6)
number_question_marks = number_question_marks(0, "number_question_marks",7)
percentage_capitalized = percentage_capitalized(-20, "percentage_capitalized",8)
frequency_of_last_chars = frequency_of_last_chars(0, "frequency_of_last_chars",9)
number_of_vowels_in_words = number_of_vowels_in_words(0, "number_of_vowels_in_words",10)
number_of_punctuation_in_words = number_of_punctuation_in_words(0, "number_of_punctuation_in_words",11)
keras_embedings = keras_embedings(-30, "keras_embedings",12)
number_of_consonants_in_words = number_of_consonants_in_words(0, "number_of_consonants_in_words",13)
bad_words = bad_words(0, "bad_words",14)
number_of_capitals_in_words = number_of_capitals_in_words(100,"number_of_capitals_in_words",15)
char_stats1 = char_stats1(0,"char_stats1",16)
char_stats2 = char_stats2(0,"char_stats2",17)
number_happy_emoticons = number_happy_emoticons(0, "number_happy_emoticons",18)
number_sad_emoticons = number_sad_emoticons(0, "number_sad_emoticons",19)
number_happy_emoticons_count = number_happy_emoticons_count(0, "number_happy_emoticons_count",20)
number_sad_emoticons_count = number_sad_emoticons_count(0, "number_sad_emoticons_count",21)
