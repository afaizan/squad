import json

filename = "../train-v1.1.json"

jso = open(filename)
d = json.load(jso)

train = []

data = d['data']

for topic in data:
	train += [{'context': para['context'],
				'id': qa['id'],
				'ques': qa['question'],
				'answer': qa['answers'][0]['text'],
				'answer_start': qa['answers'][0]['answer_start'],
				'answer_end': qa['answers'][0]['answer_start'] + len(qa['answers'][0]['text'])-1,
				'topic': topic['title']}
				for para in topic['paragraphs']
				for qa in para['qas']]

with open('../train.json','w') as fd:
	json.dump(train,fd)