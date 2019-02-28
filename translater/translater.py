import numpy as np
import pandas 
import pickle
import nltk
import tensorflow as tf


class Translater():

	def __init__(self, learning_rate = 0.001, batch_size=128, epochs=13, 
	num_layers= 3, rnn_size = 128, encoding_embedding_size = 200, 
	decoding_embedding_size =200, keep_prob = 0.5, display_step = 300,
	 save_path='checkpoints/dev'):

		self.learning_rate = learning_rate
		self.epochs = epochs
		self.batch_size = batch_size
		self.num_layers = num_layers
		self.rnn_size = rnn_size 
		self.keep_prob = keep_prob
		self.encoding_embedding_size = encoding_embedding_size
		self.decoding_embedding_size = decoding_embedding_size
		self.display_step = display_step
		self.save_path = save_path
		self.tf_graph = tf.Graph()
		self.tf_session = tf.Session()
		self.training_logits = None
		self.inference_logits = None
		self.cost = None
		self.train_op = None
		self.input_data = None
		self.targets = None
		self.target_sequence_length = None 
		self.max_target_sequence_length = None

	def enc_dec_model_inputs(self):
		"""
		encoder and decoder model's inputs
		"""
		inputs = tf.placeholder(tf.int32, [None,None], name="inputs")
		targets = tf.placeholder(tf.int32, [None, None], name="targets")

		target_sequence_length = tf.placeholder(tf.int32, [None], name="target_sequence_length")
		max_target_length = tf.reduce_max(target_sequence_length)

		return inputs, targets, target_sequence_length, max_target_length

	def hyperparam_inputs(self):
		lr_rate = tf.placeholder(tf.float32, name = "lr_rate")
		keep_prob = tf.placeholder(tf.float32, name = "keep_prob")

		return lr_rate, keep_prob


	def process_decoder_input(self, target_data, target_vocab_to_int, batch_size):
		"""
		add 'go' token in front of each target data
		"""
		go_id = target_vocab_to_int['<GO>']
		after_slice = tf.strided_slice(target_data, [0,0], [batch_size, -1], [1,1])
		after_concat= tf.concat([tf.fill([batch_size, 1], go_id), after_slice],1)
		return after_concat

	def encoding_layer(self, rnn_inputs, source_vocab_size) :

		"""
		return tuple : (rnn_outputs, rnn_states)
		"""
		embed = tf.contrib.layers.embed_sequence(rnn_inputs, 
			vocab_size= source_vocab_size, embed_dim=self.encoding_embedding_size)

		stacked_cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(
			tf.contrib.rnn.LSTMCell(self.rnn_size), self.keep_prob) for _ in range(self.num_layers)])

		outputs, states = tf.nn.dynamic_rnn(stacked_cells, embed, dtype = tf.float32)

		return outputs, states 


	def decoding_layer_train(self, encoder_state, dec_cell, dec_embed_input, 
		target_sequence_length, max_summary_length, output_layer):
		"""
		create a traning process in decoding layer :
		return BasicDecoderOutput containing training logits and sample_id
		"""
		dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, output_keep_prob = self.keep_prob)
		
		# for only input layer 
		helper = tf.contrib.seq2seq.TrainingHelper(dec_embed_input, target_sequence_length)

		decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, helper, encoder_state, output_layer)		

		# unrolling the decoder layer
		outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, 
			impute_finished=True, maximum_iterations = max_summary_length)

		return outputs

	def decoding_layer_infer(self, encoder_state, dec_cell, dec_embeddings, 
		start_of_sequence_id, end_of_sequence_id, max_target_sequence_length, 
		vocab_size, output_layer):
		"""
		creating an inference process in decoding layer 
		return BasicDecoderOutput containing inference logits and sample_id
		"""

		dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, output_keep_prob = self.keep_prob)

		helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings, 
			tf.fill([self.batch_size], start_of_sequence_id), end_of_sequence_id)

		decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, helper, encoder_state, 
			output_layer)

		outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True, 
			maximum_iterations = max_target_sequence_length)

		return outputs


	def decoding_layer(self, dec_input, encoder_state, target_sequence_length, 
		max_target_sequence_length, target_vocab_to_int, 
		target_vocab_size):
		"""
		create decoding layer
		return : Tuple of (Training BasicDecoderOutput, inference BasicDecoderOutput)
		"""

		target_vocab_size = len(target_vocab_to_int)
		
		dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, self.decoding_embedding_size]))

		dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

		cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(self.rnn_size) for _ in range(self.num_layers)])


		with tf.variable_scope("decode") :

			output_layer = tf.layers.Dense(target_vocab_size)

			train_output = self.decoding_layer_train(encoder_state, cells,
			 dec_embed_input, target_sequence_length, max_target_sequence_length, 
			 output_layer)


		with tf.variable_scope("decode", reuse=True) :

			infer_output = self.decoding_layer_infer(encoder_state, cells, dec_embeddings ,
				target_vocab_to_int['<GO>'], target_vocab_to_int['<EOS>'], 
				max_target_sequence_length, target_vocab_size, output_layer)


		return (train_output, infer_output)



	def seq2seq_model(self, input_data, target_data, target_sequence_length, 
		max_target_sequence_length, source_vocab_size, 
		target_vocab_size, target_vocab_to_int):
		"""
		Build the seq2seq model
		return tuple of (training BasicDecoderOutput, inference BasicDecoderOutput)
		"""

		enc_outputs, enc_states = self.encoding_layer(input_data, source_vocab_size)

		dec_input = self.process_decoder_input(target_data, target_vocab_to_int, self.batch_size)


		train_output , infer_output = self.decoding_layer(dec_input, enc_states, 
			target_sequence_length, max_target_sequence_length, target_vocab_to_int, target_vocab_size)


		return train_output, infer_output




	def build_model(self, source_int_text, target_int_text, source_vocab_to_int, 
		target_vocab_to_int):
		"""
		Build the model : compute loss function and add optimizer
		"""
		max_target_sentence_length = max([len(sentence) for sentence in source_int_text])

		self.input_data, self.targets, self.target_sequence_length, self.max_target_sequence_length = self.enc_dec_model_inputs()

		self.lr, self.keep_probability = self.hyperparam_inputs()

		self.train_logits, self.inference_logits = self.seq2seq_model(tf.reverse(self.input_data, [-1]), 
			self.targets, self.target_sequence_length, self.max_target_sequence_length, 
			len(source_vocab_to_int), len(target_vocab_to_int), target_vocab_to_int)

		self.training_logits = tf.identity(self.train_logits.rnn_output, name="logits")
		self.inference_logits = tf.identity(self.inference_logits.sample_id, name="predictions")


		masks = tf.sequence_mask(self.target_sequence_length, self.max_target_sequence_length, 
			dtype=tf.float32, name='masks')

		with tf.name_scope("optimization") :
			# Loss Function- weighted softmax cross entropy
			self.cost = tf.contrib.seq2seq.sequence_loss(self.training_logits, self.targets, masks)

			# optimizer
			optimizer = tf.train.AdamOptimizer(self.lr)


			# gradients clipping 
			gradients = optimizer.compute_gradients(self.cost)

			capped_gradients = [(tf.clip_by_value(grad,-1., 1.), var) for grad, var in gradients if grad is not None]

			self.train_op = optimizer.apply_gradients(capped_gradients)


	def pad_sentence_batch(self, sentence_batch, pad_int):
		"""
		pad sentence with <pad> so each sentence of a batch has the same length
		"""
		max_sentence = max([len(sentence) for sentence in sentence_batch])

		return [sentence + [pad_int] *(max_sentence - len(sentence)) for sentence in sentence_batch]


	def get_batches(self, sources, targets, batch_size, source_pad_int, target_pad_int):
		"""
		batch targets, sources, and the lengths of their sentences together
		"""
		for batch_i in range(0, len(sources)//self.batch_size):
			start_i = batch_i*batch_size

			# slice the right amount of the batch
			sources_batch = sources[start_i:start_i+self.batch_size]
			targets_batch = targets[start_i:start_i+self.batch_size]

			# Pad 
			pad_sources_batch = np.array(self.pad_sentence_batch(sources_batch, source_pad_int))
			pad_targets_batch = np.array(self.pad_sentence_batch(targets_batch, target_pad_int))

			# need the lengths for the _lengths parametrs
			pad_targets_lengths = [] 
			for target in pad_targets_batch : 
				pad_targets_lengths.append(len(target))


			pad_sources_lengths = [] 
			for source in pad_sources_batch : 
				pad_sources_lengths.append(len(source))

			yield pad_sources_batch, pad_targets_batch, pad_sources_lengths, pad_targets_lengths


	def get_accuracy(self, logits, target):
		"""
		calculate the accuracy
		"""
		max_seq = max(logits.shape[1], target.shape[1])

		if max_seq - target.shape[1] :
			target = np.pad(target, [(0,0),(0,max_seq - target.shape[1])], 'constant')

		if max_seq - logits.shape[1] :
			logits = np.pad(logits, [(0,0),(0,max_seq - logits.shape[1])], 'constant')
			
		return np.mean(np.equal(target, logits))


	def train(self, source_int_text, target_int_text, source_vocab_to_int, 
		target_vocab_to_int, graph=None):
	
		g = graph if graph is not None else self.tf_graph
		with g.as_default():
			#self.build_model(source_int_text, target_int_text, source_vocab_to_int, target_vocab_to_int)
			
			# modification
			input_data, targets, target_sequence_length, max_target_sequence_length = self.enc_dec_model_inputs()

			lr, keep_probability = self.hyperparam_inputs()

			train_logits, inference_logits = self.seq2seq_model(tf.reverse(input_data, [-1]), 
			targets, target_sequence_length, max_target_sequence_length, 
			len(source_vocab_to_int), len(target_vocab_to_int), target_vocab_to_int)

			training_logits = tf.identity(train_logits.rnn_output, name="logits")
			inference_logits = tf.identity(inference_logits.sample_id, name="predictions")


			masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, 
			dtype=tf.float32, name='masks')

			with tf.name_scope("optimization") :
				# Loss Function- weighted softmax cross entropy
				cost = tf.contrib.seq2seq.sequence_loss(training_logits, targets, masks)

				# optimizer
				optimizer = tf.train.AdamOptimizer(lr)


				# gradients clipping 
				gradients = optimizer.compute_gradients(cost)

				capped_gradients = [(tf.clip_by_value(grad,-1., 1.), var) for grad, var in gradients if grad is not None]

				train_op = optimizer.apply_gradients(capped_gradients)




			# splitting data to training and validation sets 
			train_source = source_int_text[self.batch_size:]
			train_target = target_int_text[self.batch_size:]
			valid_source = source_int_text[:self.batch_size]
			valid_target = target_int_text[:self.batch_size]

			(valid_sources_batch, valid_targets_batch, valid_sources_lengths, valid_targets_lengths ) = next(
				self.get_batches(valid_source, valid_target, self.batch_size, source_vocab_to_int['<PAD>'], 
					target_vocab_to_int['<PAD>']))

			with tf.Session() as self.tf_session:
				self.tf_session.run(tf.global_variables_initializer())

				for epoch_i in range(self.epochs) :

					for batch_i, (source_batch, target_batch, sources_lengths, targets_lengths) in enumerate(
						self.get_batches(train_source, train_target, self.batch_size, source_vocab_to_int['<PAD>'], 
							target_vocab_to_int['<PAD>'])) :
						_, loss = self.tf_session.run(
							[train_op, cost], 
							{input_data : source_batch, 
							targets : target_batch, 
							lr : self.learning_rate, 
							target_sequence_length : targets_lengths, 
							keep_probability : self.keep_prob})

						

						if batch_i % self.display_step == 0 and batch_i > 0 :

							batch_train_logits = self.tf_session.run(
								inference_logits, 
								{input_data : source_batch, 
								target_sequence_length : targets_lengths, 
								keep_probability : 1.0})

							batch_valid_logits = self.tf_session.run(
								inference_logits, 
								{input_data : valid_sources_batch, 
								target_sequence_length : valid_targets_lengths, 
								keep_probability : 1.0})
							
							train_acc = self.get_accuracy(target_batch, batch_train_logits)
							valid_acc = self.get_accuracy(valid_targets_batch, batch_valid_logits)


							print("Epoch {:3} Batch {:>4}/{} - Train Accuracy: {:>6.4f}, Validation Accuracy: {:>6.4f}, Loss:{:>6.4f}".format(
								epoch_i, batch_i, len(source_int_text) // self.batch_size, train_acc, valid_acc, loss))

				# save model
				saver = tf.train.Saver()
				saver.save(self.tf_session, self.save_path)
				print("model trained and saved ")
		self.save_params(self.save_path)
		
		

	def save_params(self, params):
		with open('params.p', mode='wb') as out_file :
			pickle.dump(params, out_file)


	def sentence_to_seq(self, sentence, vocab_to_int) :
		results = []
		for word in sentence.split(" ") :
			if word in vocab_to_int :
				results.append(vocab_to_int[word])
			else :
				results.append(vocab_to_int['<UNK>'])
		return results


	def predict(self, sentence_to_transalte, source_vocab_to_int, 
		source_int_to_vocab, target_vocab_to_int, target_int_to_vocab, load_path) :
		"""
		take a sentence and return her traduction en the other language
		"""
		translate_sentence = self.sentence_to_seq(sentence_to_transalte, source_vocab_to_int)
		loaded_graph = tf.Graph()
		with tf.Session(graph = loaded_graph) as sess :
			# load saved model
			loader = tf.train.import_meta_graph(load_path + '.meta')
			loader.restore(sess, load_path)
			input_data = loaded_graph.get_tensor_by_name('inputs:0')
			logits = loaded_graph.get_tensor_by_name('predictions:0')
			target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')
			keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

			translate_logits = sess.run(logits, {
				input_data : [translate_sentence]*self.batch_size, 
				target_sequence_length : [len(translate_sentence)*2]*self.batch_size, 
				keep_prob : 1.0})[0]

		print("Input")
		print("Words_id  : {}".format([i for i in translate_sentence]))
		print("English words : {}".format([source_int_to_vocab[i] for i in translate_sentence]))

		print(" Predictions : ")
		print("Words_id  : {}".format([i for i in translate_logits]))
		print("Frensh words : {}".format(" ".join([target_int_to_vocab[i] for i in translate_logits])))
