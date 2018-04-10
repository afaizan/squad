import json

filename = "../train-v1.1.json"

jso = open(filename)
d = json.load(jso)

paragraph = []

for i in range(len(d['data'])):
	for j in range(len(d['data'][i]['paragraphs'])):
		paragraph.append(d['data'][i]['paragraphs'][j]['context'])

print(len(paragraph))

for i in range(10):
	print(paragraph[i])
	print("==========================================================================\n")
