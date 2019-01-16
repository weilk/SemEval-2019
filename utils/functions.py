import pandas as pd
import numpy as np
import csv
from keras import backend as K
from collections import Counter
import numpy as np 
import pandas as pd 

from sklearn.feature_extraction.text import CountVectorizer
from keras.layers import Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential,clone_model
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras import optimizers
from keras.callbacks import ModelCheckpoint
import tensorflow as tf

import keras.backend as K

from utils import *


list_boosting = []

output_emocontext = ["label"]
output_offenseval = ["suba", "subb", "subc"]

input_emocontext = ["id","turn1", "turn2", "turn3"]
input_offenseval = ["text"]

def parse_file(file_path, file_type):
	output_dict = dict()
	output_names = []
	if file_type == "EmoContext":
		output_names = input_emocontext + output_emocontext
	elif file_type == "OffensEval":
		output_names = input_offenseval + output_offenseval
	else:
		raise NameError("Invalid file type. Options: EmoContext, OffensEval")

	with open(file_path, newline='\n', encoding='utf8') as csvfile:
		csvreader = csv.reader(csvfile, delimiter='\t')
		counter = 0
		for row in csvreader:		
			if file_type == "EmoContext" and counter == 0:
				counter = 1
				continue
			for i in range(len(row)):
				if output_names[i] not in output_dict.keys():
					output_dict[output_names[i]] = list()
				output_dict[output_names[i]].append(row[i])

	return pd.DataFrame(output_dict)	


import os
from keras.models import model_from_json
import numpy
from keras.models import Sequential, Model
from keras.layers import Concatenate, concatenate, Dense

models_dir = "models"

def save_model(model):
	model_json = model.to_json()
	if not os.path.exists(models_dir):
		os.makedirs(models_dir)
	model_path = os.path.join(models_dir, "{}_model".format(model.name))
	models_json_path = model_path + ".json"
	with open(models_json_path, "w") as json_file:
		json_file.write(model_json)

	models_weights_path = model_path + ".weights"
	model.save_weights(models_weights_path)


def load_models():
	models = os.listdir(models_dir)
	models_jsons = [x for x in models if ".json" in x]
	models_jsons = [os.path.join(models_dir, x) for x in models_jsons]
	models_weights = [x.split(".json")[0].strip() + ".weights" for x in models_jsons]
	final_loaded_models = []
	for model_json in models_jsons:
		# load json and create model
		json_file = open(model_json, 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json)
		# load weights into new model
		weights_name = model_json.split(".json")[0].strip() + ".weights"
		loaded_model.load_weights(weights_name)
		for i, layer in enumerate(loaded_model.layers):
			loaded_model.layers[i].name = "{}_{}".format(model_json.rsplit("\\", 1)[1].strip(), loaded_model.layers[i].name)
			loaded_model.layers[i].trainable = False
		#loaded_model.get_layer("embedding_1_input").name = model_json.rsplit("\\", 1)[1].strip() + "embedding_1_input"
			#print("{}_{}".format(model_json.rsplit("\\", 1)[1].strip(), loaded_model.layers[i].name))
		#loaded_model.input.name = "{}_{}".format(model_json.rsplit("\\", 1)[1].strip(), loaded_model.input.name)
		print("Loaded model {} from disk".format(model_json))
		final_loaded_models.append(loaded_model)

	return final_loaded_models

def create_final_layers(final_loaded_models):
	models_layers = [model.layers[-1].output for model in final_loaded_models]
	input_layers = [model.input for model in final_loaded_models]
	new_input = []
	for inp in input_layers:
		try:
			ceva = len(inp)
			for i in inp:
				new_input.append(i)
		except:
			new_input.append(inp)
	merged_layer = concatenate(models_layers)
	return merged_layer, new_input

def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)