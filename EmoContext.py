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

features = []
pp=[(make_lower_case,["turn1","turn2","turn3"]),(one_hot_encode,["label"])]
fe=[(number_of_words,["turn1","turn2","turn3"]),(number_of_capitalized_words,["turn1","turn2","turn3"]),(number_of_elongated_words,["turn1","turn2","turn3"]),(number_negation_words,["turn1","turn2","turn3"]),(frequency_of_last_chars,["turn1", "turn2", "turn3"])]

data_object = data(raw=emocontext_DataFrame,pp=pp,fe=fe)
msk = np.random.rand(len(data_object.D)) < 0.8
data_object.D = data_object.D.drop(["label"],axis=1)
output_emocontext.remove("label")
data_object.D = data_object.D.drop(["turn1","turn2","turn3","id"],axis=1)
simple_MLP.train(data_object.D[msk])
simple_MLP.test(data_object.D[~msk])

# docker build -t simi2525/ml-env:cpu -f Dockerfile.cpu .
# docker run -it -p 8888:8888 -p 6006:6006  -v ${PWD}/jupyter_notebook_config.py:/root/.jupyter/jupyter_notebook_config.py -v ${PWD}:"/root/SemEval-2019" simi2525/ml-env:cpu
# cd SemEval-2019
# jupyter notebook --allow-root