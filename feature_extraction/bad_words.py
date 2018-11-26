#https://www.cs.cmu.edu/~biglou/resources/bad-words.txt

from classes import feature_extraction
from nltk.tokenize import word_tokenize
import string

banned_words = ["abo","bat","bitch","blonde","boy","brave","brit","brown","brownie","cabbage","cac","cat","chen","cholo","cookie","crip","cunt","fag","faggot","female","gin","goddamn","iron","leb","little","lop","male","mong","motherfucker","ned","nig","nigga","posh","potato","red","retard","retarded","tard","wood","whore", "stupid", "hate", "dumbass", "disappointed", "fool", "fucked", "asshole", "bullshit", "sorry", "shame", "bastard", "pig", "bitchy", "bloody", "brainless", "cheater", "cock", "condoms", "crack", "coward", "cowardly", "demotivated", "disaster", "discarded", "disgusting", "disturb", "doomed", "harsh", "horrible", "jerk", "judgmental", "lesbian", "miserable", "motherfuck", "orphanage"]

class bad_words(feature_extraction):
    def run(self,D,columns):
        def custom_function(x, word):
            target = ""
            for column in columns:
                target += "{} ".format(x[column])

            counter = 0
            if word in target:
                counter = 1

            return counter

        for word in banned_words:
            D["number_of_bad_words_{}".format(word)] = D.apply(lambda x: custom_function(x, word), axis=1)