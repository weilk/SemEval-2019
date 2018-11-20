h = open("___", "r")
c = h.readlines()
h.close()

h = open("_list_", "w")
for i in c:
	if len(i) >= 4:
		if len(i.strip().split(" ")) == 1:
			h.write("\"{}\",".format(i.strip().lower()))

h.close()