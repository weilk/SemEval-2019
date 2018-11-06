import pandas
import csv

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

	for item in output_names:
		output_dict[item] = list()

	with open(file_path, newline='\n', encoding='utf8') as csvfile:
		csvreader = csv.reader(csvfile, delimiter='\t')
		counter = 0
		for row in csvreader:		
			if file_type == "EmoContext" and counter == 0:
				counter = 1
				continue
			for i in range(len(row)):
				output_dict[output_names[i]].append(row[i])

	return pandas.DataFrame(output_dict)	

if __name__ == "__main__":
	emocontext_DataFrame = parse_file(r"D:\Facultate_Master\TAIP\SemEval-2019\raw_data\EmoContext\train.txt", "EmoContext")
	offenseval_DataFrame = parse_file(r"D:\Facultate_Master\TAIP\SemEval-2019\raw_data\OffensEval\offenseval-trial.txt", "OffensEval")