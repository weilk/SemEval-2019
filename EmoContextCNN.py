from utils import *
from preprocess import *
from postprocess import *
from feature_extraction import *
from classes import data
from feature_selection import *
from model import *
from utils import * 
import numpy as np
import pandas as pd
import csv

def get_train_test_inds(y,train_proportion=0.8):
    '''Generates indices, making random stratified split into training set and testing sets
    with proportions train_proportion and (1-train_proportion) of initial sample.
    y is any iterable indicating classes of each observation in the sample.
    Initial proportions of classes inside training and 
    testing sets are preserved (stratified sampling).
    '''

    y=np.array(y)
    train_inds = np.zeros(len(y),dtype=bool)
    test_inds = np.zeros(len(y),dtype=bool)
    values = np.unique(y)
    for value in values:
        value_inds = np.nonzero(y==value)[0]
        np.random.shuffle(value_inds)
        n = int(train_proportion*len(value_inds))

        train_inds[value_inds[:n]]=True
        test_inds[value_inds[n:]]=True

    return train_inds,test_inds



emocontext_DataFrame = functions.parse_file(r"raw_data/EmoContext/train.txt", "EmoContext")
emocontext_DataFrame_Test = functions.parse_file(r"raw_data/EmoContext/devwithoutlabels.txt", "EmoContext")

features = []

pp=[
    (eliminate_stop_words,["turn1","turn2","turn3"]),
    (replace_negation_words,["turn1","turn2","turn3"]),
    # (one_hot_encode,["label"]),
    (spellingcheck,["turn1","turn2","turn3"]),
    (embed_200, ["turn1","turn2","turn3"]),
    # TODO - remove punctuation
    # TODO - remove emoticons
    # TODO - words_not_found after spell checking
]

fe=[
    # TODO feature for punctuation
    # TODO feature for interjections
    # TODO feature for double/multiple vowels such as: "Woooooooooooooooow" or "greeeeeeeeeeeeat"
    # TODO positive elongated words | use semantic detection API/ sentiment analysis (research)
    # TODO negative elongated words |


    # (number_of_words,["turn1","turn2","turn3"]),
    # (number_of_capitalized_words,["turn1","turn2","turn3"]),
    # (number_of_elongated_words,["turn1","turn2","turn3"]),
    # (number_negation_words,["turn1","turn2","turn3"]),
    # (number_boosting_words,["turn1","turn2","turn3"]),
    # (number_exclamation_marks,["turn1","turn2","turn3"]),
    # (number_question_marks,["turn1","turn2","turn3"]),
    # (number_happy_emoticons,["turn1","turn2","turn3"]),
    # (number_sad_emoticons,["turn1","turn2","turn3"]),
    # (number_happy_emoticons_count,["turn1","turn2","turn3"]),
    # (number_sad_emoticons_count,["turn1","turn2","turn3"]),
    # (number_of_punctuation_in_words,["turn1", "turn2", "turn3"]),
    # (number_of_capitals_in_words,["turn1", "turn2", "turn3"]),
    # (number_of_vowels_in_words,["turn1", "turn2", "turn3"]),
    # (char_stats1,["turn1", "turn2", "turn3"]),
    # (number_of_consonants_in_words,["turn1", "turn2", "turn3"]),
    # (bad_words,["turn1", "turn2", "turn3"]),
    # (char_stats2,["turn1", "turn2", "turn3"]),
]
postp=[
    # (extract_redundant_words,["turn1", "turn2", "turn3"])
]

fs = [
	#(information_gain,["embedding_200_turn1","embedding_200_turn2","embedding_200_turn3"])
]

data_object = data(raw=emocontext_DataFrame,pp=pp,fe=fe,postp=postp,fs=fs)

trimping = [("others",1.0),("angry",1.0),("happy",1.0),("sad",1.0)]
aux = pd.DataFrame()
for x in trimping:
    aux = aux.append(data_object.D[(data_object.D['label'] == x[0])].sample(frac = x[1]))
data_object.D = aux.sample(frac=1.0)

print([{x:data_object.D[(data_object.D['label'] == x)].shape[0]} for x in ["happy","sad","angry","others"]])

trainIdx, validationIdx = get_train_test_inds(data_object.D["label"],0.8)

print(data_object.D.columns)
print(output_emocontext)
# data_object.D = data_object.D.drop(["label","id"],axis=1)
turns = data_object.D[['turn1','turn2','turn3']]
data_object.D = data_object.D.drop(['turn1','turn2','turn3', 'id'],axis=1)
# output_emocontext.remove("label")

print(data_object.D.columns)



print(data_object.D.shape)
print(len([x for x in trainIdx if x]))
model = cnn_emb("cnn_emb")
model.train(data_object.D,
            trainIdx,
            validationIdx,
            embedding_matrix().build_matrix(turns[trainIdx],
                    ["turn1", "turn2", "turn3"],
                    load=False),#, cache_file="matrix"),
            load=False)


# data_object = data(raw=emocontext_DataFrame_Test,pp=pp,fe=fe,postp=postp,fs=fs,test=True)
# data_object.D = data_object.D.drop(["id"],axis=1)
# predicted = model.forward_pass(data_object.D)
print("predicting")
decode = {"happy": 1, "sad":2, "angry":3, "others":4}
decode = {0: "happy", 1:"sad", 2:"angry", 3:"others"}

data_object = data(raw=emocontext_DataFrame_Test,pp=pp,fe=fe,postp=postp,fs=fs,test=True)

predicted = model.forward_pass(data_object.D)
pred_labels = []
predictions = []
D = pd.DataFrame(data_object._raw)
data_object.D = data_object.D.drop(["id"],axis=1)
print(D.shape)
print(D.columns)
# print(D)
# import ipdb
# ipdb.set_trace(context=10)
for predict in predicted:
    if np.argmax(predict) not in pred_labels:
        pred_labels.append(np.argmax(predict))
    predictions.append(decode[np.argmax(predict)])

from collections import Counter
print(Counter(predictions))

path="predicted_data/EmoContext/softmaxdr01.txt"
# D = D.rename(columns={D.columns.values[-1]:variable})
D['label'] = pd.Series(predictions, index=D.index)
D.to_csv(path,index=False , sep="\t")

print("predicted")  
# print(predicted)
# create_submision_file(data_object._raw,predicted)

# docker build -t simi2525/ml-env:cpu -f Dockerfile.cpu .
# docker run -it -p 8888:8888 -p 6006:6006  -v ${PWD}/jupyter_notebook_config.py:/root/.jupyter/jupyter_notebook_config.py -v ${PWD}:"/root/SemEval-2019" simi2525/ml-env:cpu
# cd SemEval-2019
# jupyter notebook --allow-root



# check output file format

# TODO spellchecking
# ordinea preprocesarilor si a feature extraction
# emoji interpretor based on the training data <- statistical analysis
# emoji lib - sentiment analysis
# pre trained sentiment analysis model
