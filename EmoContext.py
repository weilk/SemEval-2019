from utils import *
from preprocess import *
from feature_extraction import *
from classes import data
from feature_selection import *
from model import *
from utils import * 
import pandas as pd
import csv

emocontext_DataFrame = functions.parse_file(r"raw_data/EmoContext/train.txt", "EmoContext")
emocontext_DataFrame_Test = functions.parse_file(r"raw_data/EmoContext/devwithoutlabels.txt", "EmoContext")

features = []

pp=[
    (make_lower_case,["turn1","turn2","turn3"]),
    (eliminate_stop_words,["turn1","turn2","turn3"]),
    (replace_negation_words,["turn1","turn2","turn3"]),
    (one_hot_encode,["label"]),
]
fe=[
    (number_of_words,["turn1","turn2","turn3"]),
    (number_of_capitalized_words,["turn1","turn2","turn3"]),
    (number_of_elongated_words,["turn1","turn2","turn3"]),
    (number_negation_words,["turn1","turn2","turn3"]),
    (number_boosting_words,["turn1","turn2","turn3"]),
    (number_exclamation_marks,["turn1","turn2","turn3"]),
    (number_question_marks,["turn1","turn2","turn3"]),
    (keras_embedings,["turn1","turn2","turn3"]),
    (number_of_punctuation_in_words,["turn1", "turn2", "turn3"]),
]

data_object = data(raw=emocontext_DataFrame,pp=pp,fe=fe)
msk = np.random.rand(len(data_object.D)) < 0.8



print([{x:data_object.D[(data_object.D['label'] == x)].shape[0]} for x in ["happy","sad","angry","others"]])

trimping = [("others",1.0),("angry",1.0),("happy",1.0),("sad",1.0)]
aux = pd.DataFrame()
for x in trimping:
    aux = aux.append(data_object.D[(data_object.D['label'] == x[0])].sample(frac = x[1]))
data_object.D = aux.sample(frac=1.0)

print([{x:data_object.D[(data_object.D['label'] == x)].shape[0]} for x in ["happy","sad","angry","others"]])

data_object.D = data_object.D.drop(["label","id"],axis=1)
output_emocontext.remove("label")
#data_object.D = data_object.D.drop(["turn1","turn2","turn3","id"],axis=1)



model = simple_MLP("simple_MLP")
model.train(data_object.D)

pp=[
    (make_lower_case,["turn1","turn2","turn3"]),
    (eliminate_stop_words,["turn1","turn2","turn3"]),
    (replace_negation_words,["turn1","turn2","turn3"]),
]
fe=[
    (number_of_words,["turn1","turn2","turn3"]),
    (number_of_capitalized_words,["turn1","turn2","turn3"]),
    (number_of_elongated_words,["turn1","turn2","turn3"]),
    (number_negation_words,["turn1","turn2","turn3"]),
    (number_boosting_words,["turn1","turn2","turn3"]),
    (number_exclamation_marks,["turn1","turn2","turn3"]),
    (number_question_marks,["turn1","turn2","turn3"]),
    (keras_embedings,["turn1","turn2","turn3"]),
    (number_of_punctuation_in_words,["turn1", "turn2", "turn3"]),
]

data_object = data(raw=emocontext_DataFrame_Test,pp=pp,fe=fe)
data_object.D = data_object.D.drop(["id"],axis=1)
predicted = model.forward_pass(data_object.D)

create_submision_file(data_object._raw,predicted)

# docker build -t simi2525/ml-env:cpu -f Dockerfile.cpu .
# docker run -it -p 8888:8888 -p 6006:6006  -v ${PWD}/jupyter_notebook_config.py:/root/.jupyter/jupyter_notebook_config.py -v ${PWD}:"/root/SemEval-2019" simi2525/ml-env:cpu
# cd SemEval-2019
# jupyter notebook --allow-root