from utils import * 
emocontext_DataFrame = functions.parse_file(r"raw_data/EmoContext/train.txt", "EmoContext")
print(emocontext_DataFrame.columns.values)
max_words_turn1 = 0
for row in emocontext_DataFrame['turn1']:
	if len(row.split())> max_words_turn1:
		max_words_turn1 = len(row.split())

print(max_words_turn1)


max_words_turn1 = 0
for row in emocontext_DataFrame['turn2']:
	if len(row.split())> max_words_turn1:
		max_words_turn1 = len(row.split())

print(max_words_turn1)


max_words_turn1 = 0
for row in emocontext_DataFrame['turn3']:
	if len(row.split())> max_words_turn1:
		max_words_turn1 = len(row.split())

print(max_words_turn1)