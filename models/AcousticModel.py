# coding=utf-8
"""
Based on the paper:

http://arxiv.org/pdf/1601.06581v2.pdf

This model is:

acoustic RNN trained with ctc loss
"""

import tensorflow as tf
try:
    from tensorflow.models.rnn import rnn_cell, rnn
except ImportError:
    # TensorFlow >= 0.8
    from tensorflow.python.ops import rnn_cell, rnn
try:
    import tensorflow.contrib.ctc as ctc
except ImportError:
    # TensorFlow >= 0.10
    from tensorflow import nn as ctc
import util.audioprocessor as audioprocessor
import numpy as np
import time
import sys
import os
from datetime import datetime
from random import shuffle
import threading


class AcousticModel(object):
    def __init__(self, session, num_labels, num_layers, hidden_size, dropout,
                 batch_size, learning_rate, lr_decay_factor, grad_clip,
                 max_input_seq_length, max_target_seq_length, input_dim,
                 forward_only=False, tensorboard_dir=None, tb_run_name=None):
        """
        Acoustic rnn model, using ctc loss with lstm cells
        Inputs:
        session - tensorflow session
        num_labels - dimension of character input/one hot encoding
        num_layers - number of lstm layers
        hidden_size - size of hidden layers
        dropout - probability of dropping hidden weights
        batch_size - number of training examples fed at once
        learning_rate - learning rate parameter fed to optimizer
        lr_decay_factor - decay factor of the learning rate
        grad_clip - max gradient size (prevent exploding gradients)
        max_input_seq_length - maximum length of input vector sequence
        max_target_seq_length - maximum length of ouput vector sequence
        input_dim - dimension of input vector
        forward_only - whether to build back prop nodes or not
        tensorboard_dir - path to tensorboard file (None if not activated)
        """
        # Define GraphKeys for TensorBoard
        graphkey_training = tf.GraphKeys()
        graphkey_test = tf.GraphKeys()

        self.dropout = dropout
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, name='learning_rate')
        tf.scalar_summary('Learning rate', self.learning_rate, collections=[graphkey_training, graphkey_test])
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * lr_decay_factor)
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.dropout_keep_prob_lstm_input = tf.constant(self.dropout)
        self.dropout_keep_prob_lstm_output = tf.constant(self.dropout)
        self.max_input_seq_length = max_input_seq_length
        self.max_target_seq_length = max_target_seq_length
        self.tensorboard_dir = tensorboard_dir
        self.input_dim = input_dim

        # Initialize audio_processor to None
        self.audio_processor = None

        # graph inputs
        self.inputs = tf.placeholder(tf.float32,
                                     shape=[self.max_input_seq_length, None, self.input_dim],
                                     name="inputs")
        # We could take an int16 for less memory consumption but CTC need an int32
        self.input_seq_lengths = tf.placeholder(tf.int32,
                                                shape=[None],
                                                name="input_seq_lengths")

        # Define cells of acoustic model
        cell = rnn_cell.BasicLSTMCell(hidden_size, state_is_tuple=True)
        if not forward_only:
            # If we are in training then add a dropoutWrapper to the cells
            cell = rnn_cell.DropoutWrapper(cell, input_keep_prob=self.dropout_keep_prob_lstm_input,
                                           output_keep_prob=self.dropout_keep_prob_lstm_output)

        if num_layers > 1:
            cell = rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

        # build input layer
        with tf.name_scope('Input_Layer'):
            w_i = tf.Variable(tf.truncated_normal([input_dim, hidden_size], stddev=np.sqrt(2.0 / (2 * hidden_size))),
                              name="input_w")
            b_i = tf.Variable(tf.zeros([hidden_size]), name="input_b")

        # make rnn inputs
        inputs = [tf.matmul(tf.squeeze(i, squeeze_dims=[0]), w_i) + b_i
                  for i in tf.split(0, self.max_input_seq_length, self.inputs)]

        # set rnn init state to 0s
        init_state = cell.zero_state(self.batch_size, tf.float32)

        # build rnn
        with tf.name_scope('Dynamic_rnn'):
            rnn_output, self.hidden_state = rnn.dynamic_rnn(cell, tf.pack(inputs),
                                                            sequence_length=self.input_seq_lengths,
                                                            initial_state=init_state,
                                                            time_major=True, parallel_iterations=1000)

        # build output layer
        with tf.name_scope('Output_layer'):
            w_o = tf.Variable(tf.truncated_normal([hidden_size, num_labels], stddev=np.sqrt(2.0 / (2 * num_labels))),
                              name="output_w")
            b_o = tf.Variable(tf.zeros([num_labels]), name="output_b")

        # compute logits
        self.logits = tf.pack([tf.matmul(tf.squeeze(i, squeeze_dims=[0]), w_o) + b_o
                               for i in tf.split(0, self.max_input_seq_length, rnn_output)])

        # compute prediction
        self.prediction = tf.to_int32(ctc.ctc_beam_search_decoder(self.logits, self.input_seq_lengths)[0][0])

        if not forward_only:
            # Sparse tensor for corrects labels input
            self.sparse_labels = tf.sparse_placeholder(tf.int32)

            # compute ctc loss
            self.ctc_loss = ctc.ctc_loss(self.logits, self.sparse_labels,
                                         self.input_seq_lengths)
            self.mean_loss = tf.reduce_mean(self.ctc_loss)
            tf.scalar_summary('Mean loss (Training)', self.mean_loss, collections=[graphkey_training])
            tf.scalar_summary('Mean loss (Test)', self.mean_loss, collections=[graphkey_test])
            params = tf.trainable_variables()

            opt = tf.train.AdamOptimizer(self.learning_rate)
            gradients = tf.gradients(self.ctc_loss, params)
            clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                             grad_clip)
            self.update = opt.apply_gradients(zip(clipped_gradients, params),
                                              global_step=self.global_step)

            # Accuracy
            with tf.name_scope('Accuracy'):
                error_rate = tf.reduce_sum(tf.edit_distance(self.prediction, self.sparse_labels, normalize=False)) / \
                             tf.to_float(tf.size(self.sparse_labels.values))
                tf.scalar_summary('Error Rate (Training)', error_rate, collections=[graphkey_training])
                tf.scalar_summary('Error Rate (Test)', error_rate, collections=[graphkey_test])

        # TensorBoard init
        if self.tensorboard_dir is not None:
            self.train_summaries = tf.merge_all_summaries(key=graphkey_training)
            self.test_summaries = tf.merge_all_summaries(key=graphkey_test)
            if tb_run_name is None:
                run_name = datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
            else:
                run_name = tb_run_name
            self.summary_writer = tf.train.SummaryWriter(tensorboard_dir + '/' + run_name + '/', graph=session.graph)
        else:
            self.summary_writer = None

        # We need to save all variables except for the hidden_state
        # we keep it across batches but we don't need it across different runs
        # Especially when we process a one time file
        save_list = [var for var in tf.all_variables() if var.name.find('hidden_state') == -1]
        self.saver = tf.train.Saver(save_list)

    def initializeAudioProcessor(self, max_input_seq_length):
        self.audio_processor = audioprocessor.AudioProcessor(max_input_seq_length)

    @staticmethod
    def getStrLabels(_str):
        allowed_chars = "abcdefghijklmnopqrstuvwxyz .'-_"
        # Remove punctuation
        _str = _str.replace(".", "")
        _str = _str.replace(",", "")
        _str = _str.replace("?", "")
        _str = _str.replace("'", "")
        _str = _str.replace("!", "")
        _str = _str.replace(":", "")
        # add eos char
        _str += "_"
        return [allowed_chars.index(char) for char in _str]

    def getNumBatches(self, dataset):
        return len(dataset) // self.batch_size

    def step(self, session, input_feat_vecs, mfcc_lengths_batch,
             label_values_batch, label_indices_batch, forward_only=False):
        """
        Returns:
        ctc_loss, None
        """
        input_feed = {self.inputs: input_feat_vecs, self.input_seq_lengths: mfcc_lengths_batch,
                      self.sparse_labels: tf.SparseTensorValue(label_indices_batch, label_values_batch,
                                                               [self.batch_size, self.max_target_seq_length])}
        # Base output is ctc_loss and mean_loss
        output_feed = [self.ctc_loss, self.mean_loss]
        # If a tensorboard dir is configured then we add an merged_summaries operation
        if self.tensorboard_dir is not None:
            if forward_only:
                output_feed.append(self.test_summaries)
            else:
                output_feed.append(self.train_summaries)
        # If we are in training then we add the update operation
        if not forward_only:
            output_feed.append(self.update)
        outputs = session.run(output_feed, input_feed)
        if self.tensorboard_dir is not None:
            self.summary_writer.add_summary(outputs[2], self.global_step.eval())
        return outputs[0], outputs[1]

    def process_input(self, session, inputs, input_seq_lengths):
        """
        Returns:
          Translated text
        """
        input_feed = {self.inputs.name: np.array(inputs), self.input_seq_lengths.name: np.array(input_seq_lengths)}
        output_feed = [self.prediction]
        outputs = session.run(output_feed, input_feed)
        return outputs[0]

    def run_checkpoint(self, sess, checkpoint_dir, num_test_batches, dequeue_op):
        # Save the model
        checkpoint_path = os.path.join(checkpoint_dir, "acousticmodel.ckpt")
        self.saver.save(sess, checkpoint_path, global_step=self.global_step)

        # Run a test set against the current model
        if num_test_batches > 0:
            print(num_test_batches)
            mean_loss = 0.0
            for i in range(num_test_batches):
                print("On {0}th training iteration".format(i))
                input_feat_vecs, mfcc_lengths_batch, label_values_batch, label_indices_batch = \
                    self.dequeue_data(sess, dequeue_op)

                _, step_loss = self.step(sess, input_feat_vecs, mfcc_lengths_batch,
                                         label_values_batch, label_indices_batch, forward_only=True)
                mean_loss = step_loss / num_test_batches
            print("\tTest: loss %.2f" % mean_loss)
            sys.stdout.flush()

    def enqueue_data(self, coord, sess, enqueue_op, dataset, mfcc_input, mfcc_input_length, label, label_length):
        local_dataset = []

        while not coord.should_stop():
            if len(local_dataset) == 0:
                # Update the local copy of the dataset
                local_dataset = dataset[:]
                # Shuffle the local_dataset
                shuffle(local_dataset)

            [file, text] = local_dataset.pop()

            # Calculate MFCC
            mfcc_data, original_mfcc_length = self.audio_processor.process_audio_file(file)
            # Convert string to numbers
            try:
                label_data = self.getStrLabels(text)
            except:
                # Incorrect label
                print("Incorrect label for {0} ({1})".format(file, text))
                continue
            # Check sizes
            label_data_length = len(label_data)
            if (label_data_length > self.max_target_seq_length) or (original_mfcc_length > self.max_input_seq_length):
                # If either input or output vector is too long we shouldn't take this sample
                print("Warning - sample too long : {0} (input : {1} / text : {2})".format(file,
                                                                                          original_mfcc_length,
                                                                                          label_data_length))
                continue

            sess.run(enqueue_op, feed_dict={mfcc_input: mfcc_data,
                                            mfcc_input_length: original_mfcc_length,
                                            label: label_data,
                                            label_length: label_data_length})

    @staticmethod
    def dequeue_data(sess, dequeue_op):
        # Get data from the queue
        mfcc_batch, mfcc_lengths_batch, label_batch, label_lengths_batch = sess.run(dequeue_op)

        # Transpose mfcc_batch in order to get time serie as first dimension
        # [batch_size, time_serie, input_dim] ====> [time_serie, batch_size, input_dim]
        input_feat_vecs = np.swapaxes(mfcc_batch, 0, 1)

        # Label tensor must be provided as a sparse tensor.
        # Build a tensor with the values and another with the values of each label
        label_values_batch = []
        label_indices_batch = []

        for row in range(len(label_lengths_batch)):
            # Because the queue is padding, we use label_length which can be shorter than len(label)
            # We don't want to store the padding into the sparse tensor
            for column in range(label_lengths_batch[row]):
                label_indices_batch.append([row, column])
                label_values_batch.append(label_batch[row][column])

        return input_feat_vecs, mfcc_lengths_batch, label_values_batch, label_indices_batch

    def train(self, sess, test_set, train_set, steps_per_checkpoint, checkpoint_dir, max_epoch=None):
        # Create a queue for each dataset and a coordinator
        train_queue = tf.PaddingFIFOQueue(self.batch_size * 3, [tf.int32, tf.int32, tf.int32, tf.int32],
                                          shapes=[[self.max_input_seq_length, self.input_dim], [], [None], []])
        #test_queue = tf.PaddingFIFOQueue(self.batch_size * 3, [tf.int32, tf.int32, tf.int32, tf.int32],
        #                                 shapes=[[self.max_input_seq_length, self.input_dim], [], [None], []])
        coord = tf.train.Coordinator()

        # Define the enqueue operation for training data
        train_mfcc_input = tf.placeholder(tf.int32, shape=[self.max_input_seq_length, self.input_dim])
        train_mfcc_input_length = tf.placeholder(tf.int32, shape=[])
        train_label = tf.placeholder(tf.int32, shape=[None])
        train_label_length = tf.placeholder(tf.int32, shape=[])
        train_enqueue_op = train_queue.enqueue([train_mfcc_input, train_mfcc_input_length,
                                                train_label, train_label_length])
        train_dequeue_op = train_queue.dequeue_many(self.batch_size)

        # Define the enqueue operation for test data
        #TODO: loading test data is deactivated because of a conflict with the tread for train data... To debug...
        #test_mfcc_input = tf.placeholder(tf.int32, shape=[self.max_input_seq_length, self.input_dim])
        #test_mfcc_input_length = tf.placeholder(tf.int32, shape=[])
        #test_label = tf.placeholder(tf.int32, shape=[None])
        #test_label_length = tf.placeholder(tf.int32, shape=[])
        #test_enqueue_op = test_queue.enqueue([test_mfcc_input, test_mfcc_input_length,
        #                                      test_label, test_label_length])
        #test_dequeue_op = test_queue.dequeue_many(self.batch_size)

        # Create the threads
        # TODO: implement a way to slice the set in order to deploy more than one thread
        train_threads = [threading.Thread(target=self.enqueue_data,
                                          args=(coord, sess, train_enqueue_op, train_set, train_mfcc_input,
                                                train_mfcc_input_length, train_label, train_label_length))
                         for _ in range(1)]
        #test_threads = [threading.Thread(target=self.enqueue_data,
        #                                 args=(coord, sess, test_enqueue_op, test_set, test_mfcc_input,
        #                                       test_mfcc_input_length, test_label, test_label_length))
        #                for _ in range(1)]
        for t in train_threads:
            t.start()
        #for t in test_threads:
        #    t.start()

        previous_loss = 0
        no_improvement_since = 0
        num_train_batches = self.getNumBatches(train_set)

        step_time, mean_loss = 0.0, 0.0
        current_step = 1

        # Main training loop
        for _ in range(1000000):
            if coord.should_stop():
                break

            start_time = time.time()

            input_feat_vecs, mfcc_lengths_batch, label_values_batch, label_indices_batch = \
                self.dequeue_data(sess, train_dequeue_op)

            _, step_loss = self.step(sess, input_feat_vecs, mfcc_lengths_batch,
                                     label_values_batch, label_indices_batch, forward_only=False)

            # Decrease learning rate if no improvement was seen over last 6 steps
            if step_loss >= previous_loss:
                no_improvement_since += 1
                if no_improvement_since == 6:
                    sess.run(self.learning_rate_decay_op)
                    no_improvement_since = 0
                    if self.learning_rate.eval() < 1e-7:
                        # End learning process
                        break
            else:
                no_improvement_since = 0
            previous_loss = step_loss

            # Print step result
            print("Step {0} with loss {1}".format(current_step, step_loss))
            step_time += (time.time() - start_time) / steps_per_checkpoint
            mean_loss += step_loss / steps_per_checkpoint

            # Check if we are at a checkpoint
            if current_step % steps_per_checkpoint == 0:
                print("Global step %d learning rate %.4f step-time %.2f loss %.2f" %
                      (self.global_step.eval(), self.learning_rate.eval(), step_time, mean_loss))
                #num_test_batches = self.getNumBatches(test_set)
                #self.run_checkpoint(sess, checkpoint_dir, num_test_batches, test_dequeue_op)
                step_time, mean_loss = 0.0, 0.0

            current_step += 1
            if (max_epoch is not None) and (current_step > max_epoch):
                # We have reached the maximum allowed, we should exit at the end of this run
                break

            # Shuffle the train set if we have done a full round over it
            if current_step % num_train_batches == 0:
                print("Shuffling the train set")
                shuffle(train_set)

        # Ask the threads to stop.
        coord.request_stop()
        # And wait for them to actually do it.
        coord.join(train_threads)
        #coord.join(test_threads)
