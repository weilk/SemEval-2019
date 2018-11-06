from utils import *
from preprocess import *
from feature_extraction import *
from classes import data
from feature_selection import *
from model import *
from utils import * 
import pandas as pd
import csv

offenseval_DataFrame = functions.parse_file(r"raw_data\OffensEval\offenseval-trial.txt", "OffensEval")

"""
features = []
pp=[(make_lower_case,["turn1","turn2","turn3"])]
fe=[(number_of_words,["turn1","turn2","turn3"])]


data_object = data(raw=raw,pp=pp,fe=fe)

print(data_object.D["x"])
exit()
print(best_fit.run(data_object.D))
simple_MLP.train(data_object.D)
simple_MLP.test(data_object.D)
print(simple_MLP.forward_pass(data_object.D))
"""


# docker build -t simi2525/ml-env:cpu -f Dockerfile.cpu .
# docker run -it -p 8888:8888 -p 6006:6006  -v ${PWD}/jupyter_notebook_config.py:/root/.jupyter/jupyter_notebook_config.py -v ${PWD}:"/root/SemEval-2019" simi2525/ml-env:cpu
# cd SemEval-2019
# jupyter notebook --allow-root