import os.path
from utils import functions 

def get_no_words(path="max_words_per_turn"):
	if os.path.isfile(path):
		with open(path, "r") as f:
			turn1 = int(f.readline())
			turn2 = int(f.readline())
			turn3 = int(f.readline())
			return {"turn1": turn1, "turn2": turn2, "turn3": turn3}
	else:
		emocontext_DataFrame = functions.parse_file(r"raw_data/EmoContext/train.txt", "EmoContext")
		max_words_turn1, max_words_turn2, max_words_turn3 = 0, 0, 0
		for index, row in emocontext_DataFrame.iterrows():
			turn1 = len(row["turn1"].split())
			if turn1> max_words_turn1:
				max_words_turn1 = turn1
			turn2 = len(row["turn2"].split())
			if turn2> max_words_turn2:
				max_words_turn2 = turn2
			turn3 = len(row["turn3"].split())
			if turn3> max_words_turn3:
				max_words_turn3 = turn3
		with open(path, "w") as f:
			f.write("%s\n%s\n%s" % (str(max_words_turn1), str(max_words_turn2), str(max_words_turn3)))
		return {"turn1": max_words_turn1, "turn2": max_words_turn2, "turn3": max_words_turn3}
