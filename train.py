'''
Main training routine for character level speech recognizer.

This entire project is based on the model described by the paper:
http://arxiv.org/pdf/1601.06581v2.pdf
'''


import tensorflow as tf
from tensorflow.python.platform import gfile
from models.AcousticModel import *
import sys
import math
import os
import random
import time
from six.moves import xrange
import util.audioprocessor as audioprocessor
import util.dataprocessor as dataprocessor
import util.hyperparams as hyperparams
import ConfigParser
from multiprocessing import Process, Pipe
from math import floor

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("config_file", "config.ini", "Path to configuration file with hyper-parameters.")
#TODO consider consolidating into config file
flags.DEFINE_string("checkpoint_dir", "data/checkpoints/", "Directory to store/restore checkpoints")
flags.DEFINE_string("data_dir", "data/", "Path to main data directory.")
flags.DEFINE_string("raw_data_dir", "data/LibriSpeech/", "Path to unprocessed data dir.")

def main():
	hyper_params = checkGetHyperParamDic()
	print "Using checkpoint {0}".format(FLAGS.checkpoint_dir)
	print "Using hyper params: {0}".format(hyper_params)
	data_processor = dataprocessor.DataProcessor(FLAGS.data_dir,
		FLAGS.raw_data_dir, hyper_params)
	text_audio_pairs = data_processor.run()
	num_train = int(floor(hyper_params["train_frac"] * len(text_audio_pairs)))
	train_set = text_audio_pairs[num_train:]
	test_set = text_audio_pairs[:num_train]

	#setting up piplines to be able to load data async (one for test set, one for train)
	#TODO tensorflow probably has something built in for this, look into it
	parent_train_conn, child_train_conn = Pipe()
	parent_test_conn, child_test_conn = Pipe()

	with tf.Session() as sess:
		#create model
		print "Building model... (this takes a while)"
		model = createAcousticModel(sess, hyper_params)
		print "Setting up audio processor..."
		model.initializeAudioProcessor(hyper_params["max_input_seq_length"])
		print "Setting up piplines to test and train data..."
		model.setConnections(child_test_conn, child_train_conn)
		num_test_batches = model.getNumBatches(test_set)

		async_train_loader = Process(
		target=model.getBatch,
		args=(train_set, True))
		async_train_loader.start()

		step_time, loss = 0.0, 0.0
		current_step = 0
		previous_losses = []
		while True:
			#begin timer
			start_time = time.time()
			#receive batch from pipe
			step_batch_inputs = parent_train_conn.recv()
			async_train_loader.join()
			#begin fetching other batch while graph processes previous one
			async_train_loader = Process(
			target=model.getBatch,
			args=(train_set, True))
			async_train_loader.start()

			_, step_loss = model.step(sess, step_batch_inputs[0], step_batch_inputs[1],
				step_batch_inputs[2], step_batch_inputs[3],
				step_batch_inputs[4],forward_only=False)

			step_time += (time.time() - start_time) / hyper_params["steps_per_checkpoint"]
			loss += step_loss / hyper_params["steps_per_checkpoint"]
			current_step += 1

			if current_step % hyper_params["steps_per_checkpoint"] == 0:
				print ("global step %d learning rate %.4f step-time %.2f loss "
					"%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
						 step_time, loss))
				# Decrease learning rate if no improvement was seen over last 3 times.
				if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
					sess.run(model.learning_rate_decay_op)
				previous_losses.append(loss)

				checkpoint_path = os.path.join(FLAGS.checkpoint_dir, "acousticmodel.ckpt")
				model.saver.save(sess, checkpoint_path, global_step=model.global_step)
				step_time, loss = 0.0, 0.0
				#begin loading test data async
				#(uses different pipline than train data)
				async_test_loader = Process(
				target=model.getBatch,
				args=(train_set, False))
				async_test_loader.start()

				for i in range(num_test_batches):
					eval_inputs = parent_test_conn.recv()
					async_test_loader.join()
					#tell audio processor to go get another batch ready
					#while we run last one through the graph
					async_train_loader = Process(
					target=model.getBatch,
					args=(train_set, False))
					async_train_loader.start()
					_, loss = model.step(sess, eval_inputs[0], eval_inputs[1],
						eval_inputs[2], eval_inputs[3],
						eval_inputs[4],forward_only=True)
					print("\tTest: loss %.2f" % (loss))
				sys.stdout.flush()


def createAcousticModel(session, hyper_params):
	num_labels = 31
	input_dim = 123
	model = AcousticModel(num_labels, hyper_params["num_layers"],
		hyper_params["hidden_size"], hyper_params["dropout"],
		hyper_params["batch_size"], hyper_params["learning_rate"],
		hyper_params["lr_decay_factor"], hyper_params["grad_clip"],
		hyper_params["max_input_seq_length"],
		hyper_params["max_target_seq_length"],
		input_dim,
		forward_only=False)
	ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
	if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
		print "Reading model parameters from {0}".format(ckpt.model_checkpoint_path)
		model.saver.restore(session, ckpt.model_checkpoint_path)
	else:
		print "Created model with fresh parameters."
		session.run(tf.initialize_all_variables())
	return model

def checkGetHyperParamDic():
	'''
	Retrieves hyper parameter information from either config file or checkpoint
	'''
	serializer = hyperparams.HyperParameterHandler(FLAGS.checkpoint_dir)
	hyper_params = readConfigFile()
	if serializer.checkExists():
		if serializer.checkChanged(hyper_params):
			if not hyper_params["use_config_file_if_checkpoint_exists"]:
				hyper_params = serializer.getParams()
				print "Restoring hyper params from previous checkpoint..."
			else:
				new_checkpoint_dir = "{0}_hidden_size_{1}_numlayers_{2}_dropout_{3}".format(
				int(time.time()),
				hyper_params["hidden_size"],
				hyper_params["num_layers"],
				hyper_params["dropout"])
				new_checkpoint_dir = os.path.join(FLAGS.checkpoint_dir,
					new_checkpoint_dir)
				os.makedirs(new_checkpoint_dir)
				FLAGS.checkpoint_dir = new_checkpoint_dir
 				serializer = hyperparams.HyperParameterHandler(FLAGS.checkpoint_dir)
				serializer.saveParams(hyper_params)
		else:
			print "No hyper parameter changed detected, using old checkpoint..."
	else:
		serializer.saveParams(hyper_params)
		print "No hyper params detected at checkpoint... reading config file"
	return hyper_params

def readConfigFile():
	'''
	Reads in config file, returns dictionary of network params
	'''
	config = ConfigParser.ConfigParser()
	config.read(FLAGS.config_file)
	dic = {}
	acoustic_section = "acoustic_network_params"
	general_section = "general"
	dic["num_layers"] = config.getint(acoustic_section, "num_layers")
	dic["hidden_size"] = config.getint(acoustic_section, "hidden_size")
	dic["dropout"] = config.getfloat(acoustic_section, "dropout")
	dic["batch_size"] = config.getint(acoustic_section, "batch_size")
	dic["train_frac"] = config.getfloat(acoustic_section, "train_frac")
	dic["learning_rate"] = config.getfloat(acoustic_section, "learning_rate")
	dic["lr_decay_factor"] = config.getfloat(acoustic_section, "lr_decay_factor")
	dic["grad_clip"] = config.getint(acoustic_section, "grad_clip")
	dic["use_config_file_if_checkpoint_exists"] = config.getboolean(general_section,
		"use_config_file_if_checkpoint_exists")
	dic["max_input_seq_length"] = config.getint(general_section,
		"max_input_seq_length")
	dic["max_target_seq_length"] = config.getint(general_section,
		"max_target_seq_length")
	dic["steps_per_checkpoint"] = config.getint(general_section,
		"steps_per_checkpoint")
	return dic

if __name__=="__main__":
	main()
