import random
import pickle
import os

import numpy as np
import keras.utils as ku
from keras.callbacks import EarlyStopping
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.models import model_from_json
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from .language_model import LanguageModel


class LstmLanguageModel(LanguageModel):

	def __init__(self, mode, slug, base_dir, embeddings="keras", oov_token='UNK'):
		supported_embeddings = ["keras"]

		self.dir = os.path.join(base_dir, "{}_{}".format('LSTM', slug))
		if not os.path.exists(self.dir):
			os.makedirs(self.dir)

		if embeddings in supported_embeddings:
			self.embeddings = embeddings
		else:
			raise Exception(
				"Unsupported Embeddings Type: {}".format(embeddings))
		
		self.mode = mode
		self.word_count = 0
		self.slug = slug
		self.oov_token = oov_token
		self.slug = slug

	def create_model(self, lstm_hidden_cells):
		model = Sequential()
		model.add(Embedding(self.word_count, 10,
							input_length=self.max_sequence_len-1))

		for cell_count in lstm_hidden_cells[:-1]:
			model.add(LSTM(cell_count, return_sequences=True))

		model.add(LSTM(lstm_hidden_cells[-1]))
		
		model.add(Dense(self.word_count, activation='softmax'))
		
		model.compile(loss='categorical_crossentropy',
					  optimizer='adam', metrics=['accuracy'])
		
		self.model = model

	def prepare_data(self, input_data):
		tokenizer = Tokenizer(oov_token=self.oov_token)

		# basic cleanup
		data = "\n".join(input_data)
		data.replace(".", ".\n")
		data.replace("?", "?\n")
		data = data.lower().split("\n")

		# tokenization
		tokenizer.fit_on_texts(data)
		self.word_count = len(tokenizer.word_index) + 1
		# tokenizer.word_index[tokenizer.oov_token] = self.word_count

		# create input sequences using list of tokens
		input_sequences = []

		for line in data:
			token_list = tokenizer.texts_to_sequences([line])[0]

			for i in range(1, len(token_list)):
				n_gram_sequence = token_list[:i+1]
				input_sequences.append(n_gram_sequence)

		# pad sequences
		self.max_sequence_len = max([len(x) for x in input_sequences])

		input_sequences = np.array(pad_sequences(
			input_sequences, maxlen=self.max_sequence_len, padding='pre'))

		# create predictors and label
		self.predictors = input_sequences[:, :-1]
		self.label = input_sequences[:, -1]
		self.tokenizer = tokenizer

		return data

	def generator(self):
		index = 0
		sentence_list = self.predictors

		while True:
			x = []
			y = []

			for i in range(self.batch_size):
				x.append(sentence_list[index])
				y.append(ku.to_categorical(
					self.label[index], num_classes=self.word_count))
				index += 1

				if index == len(sentence_list):
					index = 0

			yield np.array(x), np.array(y)

	def train(self, inputData, BATCH_SIZE=64, lstm_hidden_cells=[150, 100], epochs=50, earlystop_patience=5):
		""" inputData -> Array of Strings
			lstm_hidden_cells -> len = number of layers, each number == hidden cells in that layer
						epochs -> number of epochs """

		self.data = self.prepare_data(inputData)
		self.create_model(lstm_hidden_cells)
		self.batch_size = BATCH_SIZE

		earlystop = EarlyStopping(
			monitor='loss', min_delta=0, patience=earlystop_patience, verbose=0, mode='auto')

		self.model.fit_generator(
			self.generator(),
			steps_per_epoch=int(len(self.predictors)/self.batch_size) + 1,
			epochs=epochs,
			verbose=1,
			callbacks=[earlystop])

		self.save_model()

	def save_model(self):
		# Set Directory for this slug
		# Have custom location flag

		with open(os.path.join(self.dir,'tokenizer.pickle'), 'wb') as handle:
			pickle.dump(self.tokenizer, handle,
						protocol=pickle.HIGHEST_PROTOCOL)

		self.model.save_weights(os.path.join(self.dir,'model_weights.h5'))

		with open(os.path.join(self.dir,'model_architecture.json'), 'w') as f:
			f.write(self.model.to_json())

	def load_model(self):
		# Set Directory for this slug
		# Have custom location flag

		with open(os.path.join(self.dir,'tokenizer.pickle'), 'rb') as handle:
			self.tokenizer = pickle.load(handle)

		with open(os.path.join(self.dir,'model_architecture.json'), 'r') as f:
			self.model = model_from_json(f.read())

		self.model.load_weights(os.path.join(self.dir,'model_weights.h5'))

	def generate_n_choices(self, n, seed_text=None):
		wi = self.tokenizer.word_index

		if not seed_text:
			print("Unconditional Text Generation: 3 Random Words to start with")
			word_options = random.sample(wi.keys(), n)
			return word_options

		print("Conditional Text Generation: on {}".format(seed_text))

		seed_token_list = self.tokenizer.texts_to_sequences([seed_text])[0]

		seed_seq = pad_sequences(
			[seed_token_list], maxlen=self.max_sequence_len-1, padding='pre')

		predicted = self.model.predict(seed_seq, verbose=0)

		top_n = predicted[0].argsort()[-n:]

		word_options = []
		for each in wi:
			if wi[each] in top_n:
				word_options.append(each)

		return seed_text, word_options

	def generate_n_words(self, n, seed_text=None):
		wi = self.tokenizer.word_index

		for t in range(n):

			if not seed_text:
				print("Unconditional Text Generation")
				a = random.choice(list(wi.keys()))
				seed_text = a
				continue

			seed_token_list = self.tokenizer.texts_to_sequences([seed_text])[0]

			seed_seq = pad_sequences(
				[seed_token_list], maxlen=self.max_sequence_len-1, padding='pre')

			predicted = self.model.predict_classes(seed_seq, verbose=0)

			for each in wi:
				if wi[each] == predicted:
					seed_text += " " + each
					break

		return seed_text
