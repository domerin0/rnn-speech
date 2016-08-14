'''
Based on the paper:

http://arxiv.org/pdf/1601.06581v2.pdf

This model is:

acoustic RNN trained with ctc loss
'''

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
from multiprocessing import Process, Pipe
import time
import sys
import os

class AcousticModel(object):
    def __init__(self, num_labels, num_layers, hidden_size, dropout,
                 batch_size, learning_rate, lr_decay_factor, grad_clip,
                 max_input_seq_length, max_target_seq_length, input_dim, forward_only=False):
        '''
        Acoustic rnn model, using ctc loss with lstm cells
        Inputs:
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
        '''
        self.dropout = dropout
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * lr_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)
        self.dropout_keep_prob_lstm_input = tf.constant(self.dropout)
        self.dropout_keep_prob_lstm_output = tf.constant(self.dropout)
        self.max_input_seq_length = max_input_seq_length
        self.max_target_seq_length = max_target_seq_length

        # Initialize data pipes to None
        self.train_conn = None
        self.test_conn = None

        # graph inputs
        self.inputs = tf.placeholder(tf.float32,
                                     shape=[self.max_input_seq_length, None, input_dim],
                                     name="inputs")
        # We could take an int16 for less memory consumption but CTC need an int32
        self.input_seq_lengths = tf.placeholder(tf.int32,
                                                shape=[None],
                                                name="input_seq_lengths")
        # Take an int16 for less memory consumption
        # max_target_seq_length should be less than 65535 (which is huge)
        self.target_seq_lengths = tf.placeholder(tf.int16,
                                                 shape=[None],
                                                 name="target_seq_lengths")

        # define cells of acoustic model
        cell = rnn_cell.DropoutWrapper(
            rnn_cell.BasicLSTMCell(hidden_size),
            input_keep_prob=self.dropout_keep_prob_lstm_input,
            output_keep_prob=self.dropout_keep_prob_lstm_output)

        if num_layers > 1:
            cell = rnn_cell.MultiRNNCell([cell] * num_layers)

        # build input layer
        w_i = tf.get_variable("input_w", [input_dim, hidden_size])
        b_i = tf.get_variable("input_b", [hidden_size])

        # make rnn inputs
        inputs = [tf.matmul(tf.squeeze(i, squeeze_dims=[0]), w_i) + b_i
                  for i in tf.split(0, self.max_input_seq_length, self.inputs)]

        # set rnn init state to 0s
        initial_state = cell.zero_state(self.batch_size, tf.float32)

        # build rnn
        rnn_output, self.hidden_state = rnn.dynamic_rnn(cell, tf.pack(inputs),
                                                        sequence_length=self.input_seq_lengths,
                                                        initial_state=initial_state,
                                                        time_major=True, parallel_iterations=1000)

        # build output layer
        w_o = tf.get_variable("output_w", [hidden_size, num_labels])
        b_o = tf.get_variable("output_b", [num_labels])

        # compute logits
        self.logits = [tf.matmul(tf.squeeze(i, squeeze_dims=[0]), w_o) + b_o
                       for i in tf.split(0, self.max_input_seq_length, rnn_output)]

        if forward_only:
            self.logits = tf.pack(self.logits)
        else:
            # graph sparse tensor inputs
            # We could take an int16 for less memory consumption but SparseTensor need an int64
            self.target_indices = tf.placeholder(tf.int64,
                                                 shape=[None, 2],
                                                 name="target_indices")
            # We could take an int8 for less memory consumption but CTC need an int32
            self.target_vals = tf.placeholder(tf.int32,
                                              shape=[None],
                                              name="target_vals")

            # setup sparse tensor for input into ctc loss
            sparse_labels = tf.SparseTensor(
                indices=self.target_indices,
                values=self.target_vals,
                shape=[self.batch_size, self.max_target_seq_length])

            # compute ctc loss
            self.ctc_loss = ctc.ctc_loss(tf.pack(self.logits), sparse_labels,
                                         self.input_seq_lengths)
            self.mean_loss = tf.reduce_mean(self.ctc_loss)
            params = tf.trainable_variables()

            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            gradients = tf.gradients(self.ctc_loss, params)
            clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                             grad_clip)
            self.update = opt.apply_gradients(zip(clipped_gradients, params),
                                              global_step=self.global_step)

        self.saver = tf.train.Saver(tf.all_variables())

    def getBatch(self, dataset, batch_pointer, is_train):
        '''
        Inputs:
          dataset - tuples of (wav file, transcribed_text)
          batch_pointer - start point in dataset from where to take the batch
          is_train - training mode (to choose which pipe to use)
        Returns:
          input_feat_vecs, input_feat_vec_lengths, target_lengths,
            target_labels, target_indices
        '''
        initial_batch_pointer = batch_pointer
        input_feat_vecs = []
        input_feat_vec_lengths = []
        target_lengths = []
        target_labels = []
        target_indices = []

        batch_counter = 0
        while batch_counter < self.batch_size:
            file_text = dataset[batch_pointer]
            batch_pointer += 1
            if batch_pointer == dataset.__len__():
                batch_pointer = 0
            assert batch_pointer != initial_batch_pointer

            # Process the audio file to get the input
            feat_vec, original_feat_vec_length = self.audio_processor.processFLACAudio(file_text[0])
            # Process the label to get the output
            labels = self.getStrLabels(file_text[1])
            # Labels len does not need to be always the same as for input, don't need padding

            # Check sizes
            if (len(labels) > self.max_target_seq_length) or (original_feat_vec_length > self.max_input_seq_length):
                # If either input or output vector is too long we shouldn't take this sample
                print("Warning - sample too long : {0} (input : {1} / text : {2})".format(file_text[0],
                      original_feat_vec_length, len(labels)))
                continue

            assert len(labels) <= self.max_target_seq_length
            assert len(feat_vec) <= self.max_input_seq_length

            # Add input to inputs matrix and unpadded or cut size to dedicated vector
            input_feat_vecs.append(feat_vec)
            input_feat_vec_lengths.append(min(original_feat_vec_length, self.max_input_seq_length))

            # Compute sparse tensor for labels
            indices = [[batch_counter, i] for i in range(len(labels))]
            target_indices += indices
            target_labels += labels
            target_lengths.append(len(labels))
            batch_counter += 1

        input_feat_vecs = np.swapaxes(input_feat_vecs, 0, 1)
        if is_train and self.train_conn is not None:
            self.train_conn.send([input_feat_vecs, input_feat_vec_lengths,
                                  target_lengths, target_labels, target_indices, batch_pointer])
        elif not is_train and self.test_conn is not None:
            self.test_conn.send([input_feat_vecs, input_feat_vec_lengths,
                                 target_lengths, target_labels, target_indices, batch_pointer])
        else:
            return [input_feat_vecs, input_feat_vec_lengths,
                    target_lengths, target_labels, target_indices, batch_pointer]

    def initializeAudioProcessor(self, max_input_seq_length):
        self.audio_processor = audioprocessor.AudioProcessor(max_input_seq_length)

    def setConnections(self):
        # setting up piplines to be able to load data async (one for test set, one for train)
        # TODO tensorflow probably has something built in for this, look into it
        parent_train_conn, self.train_conn = Pipe()
        parent_test_conn, self.test_conn = Pipe()
        return parent_train_conn, parent_test_conn

    def getCharLabel(self, char):
        '''
        char is a length 1 string
        '''
        assert len(char) == 1
        # _ will be used as eos character
        return "abcdefghijklmnopqrstuvwxyz .'_-".index(char)

    def getStrLabels(self, _str):
        allowed_chars = "abcdefghijklmnopqrstuvwxyz .'_-"
        # add eos char
        _str += "-"
        return [allowed_chars.index(char) for char in _str]

    def getNumBatches(self, dataset):
        return len(dataset) // self.batch_size

    def step(self, session, inputs, input_seq_lengths, target_seq_lengths,
             target_vals, target_indices, forward_only=False):
        '''
        Returns:
        ctc_loss, None
        ctc_loss, None
        '''
        input_feed = {}
        input_feed[self.inputs.name] = np.array(inputs)
        input_feed[self.input_seq_lengths.name] = np.array(input_seq_lengths)
        input_feed[self.target_seq_lengths.name] = np.array(target_seq_lengths)
        input_feed[self.target_indices.name] = np.array(target_indices)
        input_feed[self.target_vals.name] = target_vals
        if not forward_only:
            output_feed = [self.ctc_loss, self.update, self.mean_loss]
        else:
            output_feed = [self.ctc_loss, self.mean_loss]
        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[0], outputs[2]
        else:
            return outputs[0], outputs[1]

    def process_input(self, session, inputs, input_seq_lengths):
        '''
        Returns:
          Translated text
        '''
        input_feed = {}
        input_feed[self.inputs.name] = np.array(inputs)
        input_feed[self.input_seq_lengths.name] = np.array(input_seq_lengths)
        output_feed = [self.logits]
        outputs = session.run(output_feed, input_feed)
        return outputs[0]

    def train(self, sess, test_set, train_set, steps_per_checkpoint, checkpoint_dir):
        print("Setting up piplines to test and train data...")
        parent_train_conn, parent_test_conn = self.setConnections()

        num_test_batches = self.getNumBatches(test_set)

        train_batch_pointer = 0
        test_batch_pointer = 0

        async_train_loader = Process(
            target=self.getBatch,
            args=(train_set, train_batch_pointer, True))
        async_train_loader.start()

        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        while True:
            # begin timer
            start_time = time.time()
            # receive batch from pipe
            step_batch_inputs = parent_train_conn.recv()

            train_batch_pointer = step_batch_inputs[5]

            # begin fetching other batch while graph processes previous one
            async_train_loader = Process(
                target=self.getBatch,
                args=(train_set, train_batch_pointer, True))
            async_train_loader.start()

            _, step_loss = self.step(sess, step_batch_inputs[0], step_batch_inputs[1],
                                      step_batch_inputs[2], step_batch_inputs[3],
                                      step_batch_inputs[4], forward_only=False)

            print("Step {0} with loss {1}".format(current_step, step_loss))
            step_time += (time.time() - start_time) / steps_per_checkpoint
            loss += step_loss / steps_per_checkpoint
            current_step += 1
            if current_step % steps_per_checkpoint == 0:
                print("global step %d learning rate %.4f step-time %.2f loss %.2f" %
                      (self.global_step.eval(), self.learning_rate.eval(), step_time, loss))
                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(self.learning_rate_decay_op)
                previous_losses.append(loss)

                checkpoint_path = os.path.join(checkpoint_dir, "acousticmodel.ckpt")
                self.saver.save(sess, checkpoint_path, global_step=self.global_step)
                step_time, loss = 0.0, 0.0
                # begin loading test data async
                # (uses different pipline than train data)
                async_test_loader = Process(
                    target=self.getBatch,
                    args=(test_set, test_batch_pointer, False))
                async_test_loader.start()
                print(num_test_batches)
                for i in range(num_test_batches):
                    print("On {0}th training iteration".format(i))
                    eval_inputs = parent_test_conn.recv()
                    # async_test_loader.join()
                    test_batch_pointer = eval_inputs[5]
                    # tell audio processor to go get another batch ready
                    # while we run last one through the graph
                    if i != num_test_batches - 1:
                        async_test_loader = Process(
                            target=self.getBatch,
                            args=(test_set, test_batch_pointer, False))
                        async_test_loader.start()
                    _, loss = self.step(sess, eval_inputs[0], eval_inputs[1],
                                         eval_inputs[2], eval_inputs[3],
                                         eval_inputs[4], forward_only=True)
                print("\tTest: loss %.2f" % loss)
                sys.stdout.flush()
