import pandas as pd
import numpy as np
import csv
from keras import backend as K
from collections import Counter

encoded_output = ["label_angry","label_happy","label_others","label_sad"]

list_boosting = []

output_emocontext = ["label"]
output_offenseval = ["suba", "subb", "subc"]

input_emocontext = ["id","turn1", "turn2", "turn3"]
input_offenseval = ["text"]

#suba - NOT (notoffensive), OFF (offensive)
#subb - TIN (targeted insult), TTH (targeted threat), UNT (untargeted)
#subc - IND (individual), GRP (Group), ORG (organization or entity), OTH (other)

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


def create_submision_file(D,predictions,path="predicted_data/EmoContext/test.txt"):
	predictions = pd.DataFrame(data=predictions,columns=encoded_output)

	i = 0
	prevColumn = ""
	while True:
		variable = encoded_output[i].split("_")[0]
		if prevColumn != variable:
			prevColumn = variable
			columns = []
			while True:		
				variable = encoded_output[i].split("_")[0]
				i+=1
				if variable == prevColumn and i<len(encoded_output):
					columns.append(encoded_output[i])
				else:
					newColumn = predictions[columns].idxmax(axis=1)
					print(Counter(newColumn.values))
					D = pd.concat([D,newColumn],axis=1)
					D = D.rename(columns={D.columns.values[-1]:variable})
					D[variable] = D[variable].apply(lambda x: x.split("_")[1])
					break
		if i>=len(encoded_output):
			break

	D.to_csv(path,index=False , sep="\t")

def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())


if __name__ == "__main__":
	emocontext_DataFrame = parse_file(r"raw_data\\EmoContext\\train.txt", "EmoContext")
	offenseval_DataFrame = parse_file(r"raw_data\\OffensEval\\offenseval-trial.txt", "OffensEval")