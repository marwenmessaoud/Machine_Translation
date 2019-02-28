import copy
import numpy as np 
import os
import pandas as pd
import pickle
import translater
import tensorflow as tf 

CODES = {'<PAD>' : 0, '<EOS>' : 1, '<UNK>' : 2, '<GO>' : 3}

def load_data(path):
	input_file = os.path.join(path)
	with open(input_file, 'r', encoding='utf_8') as f : 
		data = f.read()
	print("data loaded with success")
	return data


def create_lookup_tables(text):
	# make a list of unique words 
	vocab = set(text.split())

	# starts with the special tokens
	vocab_to_int = copy.copy(CODES)
	for v_i, v in enumerate(vocab, len(CODES)):
		vocab_to_int[v] = v_i

	int_to_vocab = {v_i : v for v, v_i in vocab_to_int.items()}

	print("lookup_tables created with success")
	return vocab_to_int, int_to_vocab


def text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int):
	# emty list for converted sentences
	source_text_id = []
	target_text_id = []

	# make a list of sentences 
	source_sentences = source_text.split("\n")
	target_sentences = target_text.split("\n")

	max_source_sentence_length = max([len(sentence.split(" ")) for sentence in source_sentences])
	max_target_sentence_length = max([len(sentence.split(" ")) for sentence in target_sentences])

	print("max_target_sentence_length : ", max_target_sentence_length)
	# iterating through each sentence (sentences of source and target in the same)

	#print("length of source sentences", len(source_sentences))
	for i in range(len(source_sentences)) : 
		# extract sentence one by one 
		source_sentence = source_sentences[i]
		target_sentence = target_sentences[i]

		# make a list of tokens from the chosen sentence
		source_tokens = source_sentence.split(" ")
		target_tokens = target_sentence.split(" ")

		# empty list of converted words to index 
		source_tokens_id = []
		target_tokens_id = []

		for index, token in enumerate(source_tokens) :
			if (token !="") :
				source_tokens_id.append(source_vocab_to_int[token])

		for index, token in enumerate(target_tokens) :
			if (token !="") :
				target_tokens_id.append(target_vocab_to_int[token])


		# put <EOS> at the end of the chosen target sentence(when stop creatinf sequence)
		target_tokens_id.append(target_vocab_to_int['<EOS>'])

		# add each converted sentences in the final list
		source_text_id.append(source_tokens_id)
		target_text_id.append(target_tokens_id)

		#print("source {} : target {}".format(source_text_id[i], target_text_id[i]))

	print("length of source text id", len(source_text_id))
	print("text converted to ids successfully")
	return source_text_id, target_text_id


def preprocess_and_save_data(source_path, target_path):
	# load original data 
	source_text = load_data(source_path)
	target_text =  load_data(target_path)

	# lower case the text
	source_text = source_text.lower()
	target_text = target_text.lower()

	# create lookup tables ofr both source and target data 
	source_vocab_to_int, source_int_to_vocab = create_lookup_tables(source_text)
	target_vocab_to_int, target_int_to_vocab = create_lookup_tables(target_text)

	# create list of sentences whose words are represneted with ids
	source_text, target_text = text_to_ids(source_text, target_text, 
		source_vocab_to_int, target_vocab_to_int)

	# save data for later use 

	pickle.dump(((source_text, target_text), 
		(source_vocab_to_int, target_vocab_to_int), 
		(source_int_to_vocab, target_int_to_vocab)), open('preprocess.p', 'wb'))

	print("data preprocessed saved with success")


def load_preprocessed_data(path_preprocess):
	with open(path_preprocess, mode='rb') as in_file :
		return pickle.load(in_file)
	
def load_params():
		with open('params.p', mode='rb') as in_file :
			return pickle.load(in_file)


path_source = "data/small_vocab_en"
path_target = "data/small_vocab_fr"

preprocess_and_save_data(path_source, path_target)

path_preprocess = "preprocess.p"

(source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ =  load_preprocessed_data(path_preprocess)
print("preprocessed data loaded with success")

save_path = "checkpoints/dev"


machine_translater = translater.Translater(learning_rate = 0.001, batch_size=128, epochs=13, 
	num_layers= 3, rnn_size = 128, encoding_embedding_size = 200, 
	decoding_embedding_size =200, keep_prob = 0.5, display_step = 300,
	 save_path=save_path)

# train the translater
machine_translater.train(source_int_text, target_int_text, source_vocab_to_int, target_vocab_to_int)

# test the translater
_, (source_vocab_to_int, target_vocab_to_int), (source_int_to_vocab, target_int_to_vocab) = load_preprocessed_data(path_preprocess)

sentence_to_transalte = "she has a red car ."

load_path = load_params()

machine_translater.predict(sentence_to_transalte, source_vocab_to_int, 
		source_int_to_vocab, target_vocab_to_int, target_int_to_vocab, load_path)
