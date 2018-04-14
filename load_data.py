import json
import numpy as np
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Input, Bidirectional, LSTM, TimeDistributed, Dense
from keras.models import Sequential, Model

filename = "../train-v1.1.json"

jso = open(filename)
d = json.load(jso)

train = []

data = d['data']

print('loading dataset...', end='')
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

print('Done!')
# with open('../train.json','w') as fd:
# 	json.dump(train,fd)

# print(train[0])
train_context = [d['context'] for d in train]
train_question = [d['ques'] for d in train]

num_train_samples = len(train_context)

max_num_words = 20000
max_sequence_length = 100
embedding_dim = 300

tokenizer = Tokenizer(num_words=max_num_words)
tokenizer.fit_on_texts(train_context)
sequences = tokenizer.texts_to_sequences(train_context)

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

data = pad_sequences(sequences, maxlen=max_sequence_length)
print('Preparing embedding matrix...', end='')

num_words = min(max_num_words, len(word_index)+1)
embedding_matrix = np.zeros((num_words, embedding_dim))
print('Done!')

print('loading glove file...', end='')
model = KeyedVectors.load_word2vec_format('../glove2word2vec.txt')
print('Done!')


for word, i in word_index.items():
    if i >= max_num_words:
        continue
    try:
    	embedding_vector = model[word]
    except:
    	embedding_vector = np.zeros(embedding_dim)
    embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(num_words,
							embedding_dim,
							weights=[embedding_matrix],
							input_length=max_sequence_length,
							trainable=False)
print('Training model.')

seq_input = Input(shape=(max_sequence_length,), dtype='int32')
embedding_sequence = embedding_layer(seq_input)

x = Bidirectional(LSTM(embedding_dim, return_sequences=True),
					input_shape=(num_train_samples, embedding_dim),
					merge_mode = 'concat')(embedding_sequence)
# x = TimeDistributed(Dense(num_words))
# layer = Lambda(lambda x:x)

model = Model(seq_input, x)
model.compile(loss='categorical_crossentropy',
				optimizer='rmsprop',
				metrics=['acc'])
print(model.summary())