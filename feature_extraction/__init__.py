from feature_extraction.number_of_words import *
from feature_extraction.number_of_capitalized_words import *
from feature_extraction.number_of_elongated_words import *
from feature_extraction.number_exclamation_marks import *
from feature_extraction.number_boosting_words import *
from feature_extraction.number_question_marks import *
from feature_extraction.number_negation_words import *
from feature_extraction.percentage_capitalized import *
from feature_extraction.keras_embedings import *
from feature_extraction.number_happy_emoticons import *
from feature_extraction.number_sad_emoticons import *

number_of_words = number_of_words(5,"number_of_words")
number_of_capitalized_words = number_of_capitalized_words(5, "number_of_capitalized_words")
number_of_elongated_words = number_of_elongated_words(0, "number_of_elongated_words")
number_negation_words = number_negation_words(0, "number_negation_words")
number_boosting_words = number_boosting_words(0, "number_boosting_words")
number_exclamation_marks = number_exclamation_marks(0, "number_exclamation_marks")
number_question_marks = number_question_marks(0, "number_question_marks")
percentage_capitalized = percentage_capitalized(-20, "percentage_capitalized")
keras_embedings = keras_embedings(-30, "keras_embedings")
number_happy_emoticons = number_happy_emoticons(0, "number_happy_emoticons")
number_sad_emoticons = number_sad_emoticons(0, "number_sad_emoticons")