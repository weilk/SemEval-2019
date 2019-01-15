import pandas as pd
import numpy as np
import csv
from keras import backend as K
from collections import Counter

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