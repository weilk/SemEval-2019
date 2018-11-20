import math

lines = [line.rstrip('\n') for line in open('train.txt', encoding="utf8")]
others = 0
happy = 0
sad = 0
angry = 0
angry_count = 0
happy_count = 0
sad_count = 0 
others_count = 0
listl = []
f = open("demofile.txt", "a")
for l in lines:
    temp = l.split()
    for el in temp:
         if "others" == temp[-1]:
          others_count = others_count + 1
         if "angry" == temp[-1]:
                angry_count = angry_count + 1
         if "sad" == temp[-1]:
          sad_count = sad_count + 1
         if "happy" == temp[-1]:
          happy_count = happy_count + 1
       
         if el==":o)": 
             if "others" == temp[-1]:
                 others = others + 1
             if "happy" == temp[-1]:
                 happy = happy + 1
             if "sad" == temp[-1]:
                 sad = sad + 1
             if "angry" == temp[-1]:
                 angry = angry + 1

print("others: {0}".format(others/others_count))
print("others: {0} {1}".format(others, others_count))
print("happy: {0}".format(happy/happy_count))
print("happy: {0} {1}".format(happy, happy_count))
print("sad: {0}".format(sad/sad_count))
print("sad: {0} {1}".format(sad, sad_count))
print("angry: {0}".format(angry/angry_count))
print("angry: {0} {1}".format(angry, angry_count))

