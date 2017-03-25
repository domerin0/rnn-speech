# coding=utf-8
"""
Based on the paper:

http://arxiv.org/pdf/1601.06581v2.pdf

And some improvements from :

https://arxiv.org/pdf/1609.05935v2.pdf

This model is:

Acoustic RNN trained with ctc loss
"""

import tensorflow as tf
from tensorflow.python.client import timeline
import numpy as np
import time
import os
from datetime import datetime
import threading
import logging

ENGLISH_CHAR_MAP = [
                    # Apostrophes with one or two letters
                    "'d", "'ll", "'m", "'nt", "'s", "s'", "'t", "'ve",
                    # Double letters first
                    'bb', 'cc', 'dd', 'ee', 'ff', 'gg', 'ii', 'kk', 'll', 'mm', 'nn',
                    'oo', 'pp', 'rr', 'ss', 'tt', 'uu', 'zz',
                    # Alphabet normal and capital
                    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                    # Apostrophe only for specific cases (eg. : O'clock)
                    "'",
                    # "end of sentence" character for CTC algorithm
                    '_'
                    ]


class AcousticModel(object):
    def __init__(self, num_layers, hidden_size, batch_size, max_input_seq_length,
                 max_target_seq_length, input_dim, normalization, language='english'):
        """
        Initialize the acoustic rnn model parameters

        Parameters
        ----------
        :param num_layers: number of lstm layers
        :param hidden_size: size of hidden layers
        :param batch_size: number of training examples fed at once
        :param max_input_seq_length: maximum length of input vector sequence
        :param max_target_seq_length: maximum length of ouput vector sequence
        :param input_dim: dimension of input vector
        :param normalization: boolean indicating whether or not to normalize data in a input batch
        :param language: the language of the speech
        """
        # Store model's parameters
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.max_input_seq_length = max_input_seq_length
        self.max_target_seq_length = max_target_seq_length
        self.input_dim = input_dim
        self.normalization = normalization

        # Set language
        if language == 'english':
            self.char_map = ENGLISH_CHAR_MAP
            self.num_labels = len(self.char_map)
        else:
            raise ValueError("Invalid parameter 'language' for method '__init__'")

        # Initialize thread management
        self.thread_lock = threading.Lock()
        self.coord = None

        # Create object's variables for tensorflow ops
        self.rnn_state_zero_op = None
        self.saver_op = None

        # Create object's variable for result output
        self.prediction = None

        # Create object's variables for placeholders
        self.input_keep_prob_ph = self.output_keep_prob_ph = None
        self.inputs_ph = self.input_seq_lengths_ph = None

        # Create object's variables for training
        self.input_keep_prob = self.output_keep_prob = None
        self.queue = None
        self.queue_input_ph = self.queue_input_length_ph = self.queue_label_ph = None
        self.enqueue_op = None
        self.global_step = None
        self.learning_rate_var = None
        # Create object variables for tensorflow training's ops
        self.learning_rate_decay_op = None
        self.accumulated_mean_loss = self.acc_mean_loss_op = self.acc_mean_loss_zero_op = None
        self.accumulated_error_rate = self.acc_error_rate_op = self.acc_error_rate_zero_op = None
        self.mini_batch = self.increase_mini_batch_op = self.mini_batch_zero_op = None
        self.acc_gradients_zero_op = self.accumulate_gradients_op = None
        self.train_step_op = None

        # Create object's variables for tensorboard
        self.tensorboard_dir = None
        self.timeline_enabled = False
        self.train_summaries_op = None
        self.test_summaries_op = None
        self.summary_writer_op = None

        # Create object's variables for status checking
        self.rnn_created = False

    def create_forward_rnn(self, with_input_queue=False):
        """
        Create the forward-only RNN

        Parameters
        -------
        :return: the logits
        """
        if self.rnn_created:
            logging.fatal("Trying to create the acoustic RNN but it is already.")

        if with_input_queue:
            # Create a queue
            inputs, input_seq_lengths, _, _ = self._create_input_queue()
        else:
            # Set placeholders for input
            self.inputs_ph = tf.placeholder(tf.float32, shape=[self.max_input_seq_length, None, self.input_dim],
                                            name="inputs_ph")

            self.input_seq_lengths_ph = tf.placeholder(tf.int32, shape=[None], name="input_seq_lengths_ph")
            inputs = self.inputs_ph
            input_seq_lengths = self.input_seq_lengths_ph

        # Build the RNN
        logits, self.prediction, self.rnn_state_zero_op, _, _ = self._build_base_rnn(inputs, input_seq_lengths, True)
        return logits

    def create_training_rnn(self, input_keep_prob, output_keep_prob, grad_clip, learning_rate, lr_decay_factor):
        """
        Create the training RNN

        Parameters
        ----------
        :param input_keep_prob: probability of keeping input signal for a cell during training
        :param output_keep_prob: probability of keeping output signal from a cell during training
        :param grad_clip: max gradient size (prevent exploding gradients)
        :param learning_rate: learning rate parameter fed to optimizer
        :param lr_decay_factor: decay factor of the learning rate
        """
        if self.rnn_created:
            logging.fatal("Trying to create the acoustic RNN but it is already.")

        # Store model parameters
        self.input_keep_prob = input_keep_prob
        self.output_keep_prob = output_keep_prob

        inputs, input_seq_lengths, label_batch, _ = self._create_input_queue()

        logits, prediction, self.rnn_state_zero_op, self.input_keep_prob_ph, self.output_keep_prob_ph =\
            self._build_base_rnn(inputs, input_seq_lengths, False)

        # Object variables used as truth label for training the RNN
        # Label tensor must be provided as a sparse tensor.
        # First get indexes from non-zero positions
        idx = tf.where(tf.not_equal(label_batch, 0))
        # Then build a sparse tensor from indexes
        sparse_labels = tf.SparseTensor(idx, tf.gather_nd(label_batch, idx),
                                        [self.batch_size, self.max_target_seq_length])

        # Add the train part to the network
        self.learning_rate_var = self._add_training_on_rnn(logits, grad_clip, learning_rate, lr_decay_factor,
                                                           sparse_labels, input_seq_lengths, prediction)

    def _create_input_queue(self):
        # Create the input queue
        capacity = self.batch_size * 2
        with tf.container("PaddingFIFOQueue"):
            self.queue = tf.PaddingFIFOQueue(capacity, [tf.float32, tf.int32, tf.int32, tf.bool],
                                             [[None, self.input_dim], [], [None], [1]])

        # Define the enqueue and dequeue operations
        self.queue_input_ph = tf.placeholder(tf.float32, shape=[None, self.input_dim], name="queue_mfcc_input")
        self.queue_input_length_ph = tf.placeholder(tf.int32, shape=[], name="queue_mfcc_input_length")
        self.queue_label_ph = tf.placeholder(tf.int32, shape=[None], name="queue_label")
        self.queue_end_signal_ph = tf.placeholder(tf.bool, shape=[1], name="queue_end_signal_ph")
        self.enqueue_op = self.queue.enqueue([self.queue_input_ph, self.queue_input_length_ph,
                                              self.queue_label_ph, self.queue_end_signal_ph])
        dequeue_op = self.queue.dequeue_many(self.batch_size)

        # Get data from the queue
        mfcc_batch, mfcc_lengths_batch, label_batch, end_signal_batch = dequeue_op

        # Define an op to throw an error if the end is reached
        # If at least one file in the batch has an end_signal to True then the batch is incomplete
        assert_op = tf.assert_equal(end_signal_batch, [[False]] * self.batch_size)
        # Depend the assertion on an op that will be needed (to insure that the assert will be run)
        with tf.control_dependencies([assert_op]):
            # Object variables used as inputs for the RNN
            # Transpose mfcc_batch in order to get time serie as first dimension
            # [batch_size, time_serie, input_dim] ====> [time_serie, batch_size, input_dim]
            inputs = tf.transpose(mfcc_batch, perm=[1, 0, 2])

        return inputs, mfcc_lengths_batch, label_batch, end_signal_batch

    def _build_base_rnn(self, inputs, input_seq_lengths, forward_only=True):
        """
        Build the Acoustic RNN

        Parameters
        ----------
        :param inputs: inputs to the RNN
        :param input_seq_lengths: vector containing the length of each input from 'inputs'
        :param forward_only: whether the RNN will be used for training or not (if true then add a dropout layer)
        
        Returns
        ----------
        :returns logits: each char probability for each timestep of the input, for each item of the batch
        :returns prediction: the best prediction for the input
        :returns rnn_state_zero_op: an tensorflow op to reset the RNN internal state
        :returns input_keep_prob_ph: a placeholder for input_keep_prob of the dropout layer
                                     (None if forward_only is True)
        :returns output_keep_prob_ph: a placeholder for output_keep_prob of the dropout layer
                                      (None if forward_only is True)
        """
        # Define cells of acoustic model
        with tf.variable_scope('LSTM'):
            cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, state_is_tuple=True)

            # If building the RNN for training then add a dropoutWrapper to the cells
            input_keep_prob_ph = output_keep_prob_ph = None
            if not forward_only:
                with tf.name_scope('dropout'):
                    # Create placeholders, used to override values when running on the test set
                    input_keep_prob_ph = tf.placeholder(tf.float32)
                    output_keep_prob_ph = tf.placeholder(tf.float32)
                    cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=input_keep_prob_ph,
                                                         output_keep_prob=output_keep_prob_ph)

            # Add more layers if needed
            if self.num_layers >= 1:
                cell = tf.contrib.rnn.MultiRNNCell([cell] * self.num_layers, state_is_tuple=True)

        # Build the input layer between input and the RNN
        with tf.variable_scope('Input_Layer'):
            # TODO: review the initializer for w_i
            w_i = tf.get_variable("input_w", [self.input_dim, self.hidden_size], tf.float32,
                                  initializer=tf.random_normal_initializer())
            b_i = tf.get_variable("input_b", [self.hidden_size], tf.float32,
                                  initializer=tf.constant_initializer(0.0))

        # Apply the input layer to the network input to produce the input for the rnn part of the network
        rnn_inputs = [tf.matmul(tf.squeeze(i, axis=[0]), w_i) + b_i
                      for i in tf.split(axis=0, num_or_size_splits=self.max_input_seq_length, value=inputs)]
        # Switch from a list to a tensor
        rnn_inputs = tf.stack(rnn_inputs)

        # Add a batch normalization layer to the model if needed
        if self.normalization:
            with tf.name_scope('Normalization'):
                epsilon = 1e-3
                # Note : the tensor is [time, batch_size, input vector] so we go against dim 1
                batch_mean, batch_var = tf.nn.moments(rnn_inputs, [1], shift=None, name="moments", keep_dims=True)
                rnn_inputs = tf.nn.batch_normalization(rnn_inputs, batch_mean, batch_var, None, None,
                                                       epsilon, name="batch_norm")

        # Define a variable to store the RNN state
        with tf.variable_scope('Hidden_state'):
            hidden_state = tf.get_variable("hidden_state", [self.num_layers, 2, self.batch_size, self.hidden_size],
                                           tf.float32, initializer=tf.constant_initializer(0.0), trainable=False)
            # Arrange it to a tuple of LSTMStateTuple as needed
            l = tf.unstack(hidden_state, axis=0)
            rnn_tuple_state = tuple([tf.contrib.rnn.LSTMStateTuple(l[idx][0], l[idx][1])
                                    for idx in range(self.num_layers)])

        # Define an op to reset the hidden state to zeros
        rnn_state_zero_op = hidden_state.assign(tf.zeros_like(hidden_state))

        # Build the RNN
        with tf.name_scope('LSTM'):
            rnn_output, _ = tf.nn.dynamic_rnn(cell, rnn_inputs, sequence_length=input_seq_lengths,
                                              initial_state=rnn_tuple_state, time_major=True)

        # Build the output layer between the RNN and the char_map
        with tf.variable_scope('Output_layer'):
            # TODO: review the initializer for w_o
            w_o = tf.get_variable("output_w", [self.hidden_size, self.num_labels], tf.float32,
                                  initializer=tf.random_normal_initializer())
            b_o = tf.get_variable("output_b", [self.num_labels], tf.float32,
                                  initializer=tf.constant_initializer(0.0))

        # Compute the logits (each char probability for each timestep of the input, for each item of the batch)
        logits = tf.stack([tf.matmul(tf.squeeze(i, axis=[0]), w_o) + b_o
                          for i in tf.split(axis=0, num_or_size_splits=self.max_input_seq_length, value=rnn_output)])

        # Compute the prediction which is the best "path" of probabilities for each item of the batch
        decoded, _log_prob = tf.nn.ctc_beam_search_decoder(logits, input_seq_lengths)
        # Set the RNN result to the best path found
        prediction = tf.to_int32(decoded[0])

        return logits, prediction, rnn_state_zero_op, input_keep_prob_ph, output_keep_prob_ph

    def _add_training_on_rnn(self, logits, grad_clip, learning_rate, lr_decay_factor,
                             sparse_labels, input_seq_lengths, prediction):
        """
        Build the training add-on of the Acoustic RNN
        
        This add-on offer ops that can be used to train the network :
          * self.learning_rate_decay_op : will decay the learning rate
          * self.acc_mean_loss_op : will compute the loss and accumulate it over multiple mini-batchs
          * self.acc_mean_loss_zero_op : will reset the loss accumulator to 0
          * self.acc_error_rate_op : will compute the error rate and accumulate it over multiple mini-batchs
          * self.acc_error_rate_zero_op : will reset the error_rate accumulator to 0
          * self.increase_mini_batch_op : will increase the mini-batchs counter
          * self.mini_batch_zero_op : will reset the mini-batchs counter
          * self.acc_gradients_zero_op : will reset the gradients
          * self.accumulate_gradients_op : will compute the gradients and accumulate them over multiple mini-batchs
          * self.train_step_op : will clip the accumulated gradients and apply them on the RNN

        Parameters
        ----------
        :param logits: the output of the RNN before the beam search
        :param grad_clip: max gradient size (prevent exploding gradients)
        :param learning_rate: learning rate parameter fed to optimizer
        :param lr_decay_factor: decay factor of the learning rate
        :param sparse_labels: the labels in a sparse tensor
        :param input_seq_lengths: vector containing the length of each input from 'inputs'
        :param prediction: the predicted label given by the RNN

        Returns
        -------
        :returns: tensorflow variable keeping the current learning rate
        """
        # Define a variable to keep track of the learning process step
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # Define the variable for the learning rate
        learning_rate_var = tf.Variable(float(learning_rate), trainable=False, name='learning_rate')
        # Define an op to decrease the learning rate
        self.learning_rate_decay_op = learning_rate_var.assign(tf.multiply(learning_rate_var, lr_decay_factor))

        # Compute the CTC loss between the logits and the truth for each item of the batch
        with tf.name_scope('CTC'):
            ctc_loss = tf.nn.ctc_loss(sparse_labels, logits, input_seq_lengths)

            # Compute the mean loss of the batch (only used to check on progression in learning)
            # The loss is averaged accross the batch but before we take into account the real size of the label
            mean_loss = tf.reduce_mean(tf.truediv(ctc_loss, tf.to_float(input_seq_lengths)))

            # Set an accumulator to sum the loss between mini-batchs
            self.accumulated_mean_loss = tf.Variable(0.0, trainable=False)
            self.acc_mean_loss_op = self.accumulated_mean_loss.assign_add(mean_loss)
            self.acc_mean_loss_zero_op = self.accumulated_mean_loss.assign(tf.zeros_like(self.accumulated_mean_loss))

        # Compute the error between the logits and the truth
        with tf.name_scope('Error_Rate'):
            error_rate = tf.reduce_mean(tf.edit_distance(prediction, sparse_labels, normalize=True))

            # Set an accumulator to sum the error rate between mini-batchs
            self.accumulated_error_rate = tf.Variable(0.0, trainable=False)
            self.acc_error_rate_op = self.accumulated_error_rate.assign_add(error_rate)
            self.acc_error_rate_zero_op = self.accumulated_error_rate.assign(tf.zeros_like(self.accumulated_error_rate))

        # Count mini-batchs
        with tf.name_scope('Mini_batch'):
            # Set an accumulator to count the number of mini-batchs in a batch
            # Note : variable is defined as float to avoid type conversion error using tf.divide
            self.mini_batch = tf.Variable(0.0, trainable=False)
            self.increase_mini_batch_op = self.mini_batch.assign_add(1)
            self.mini_batch_zero_op = self.mini_batch.assign(tf.zeros_like(self.mini_batch))

        # Compute the gradients
        trainable_variables = tf.trainable_variables()
        with tf.name_scope('Gradients'):
            opt = tf.train.AdamOptimizer(learning_rate_var)
            gradients = opt.compute_gradients(ctc_loss, trainable_variables)

            # Define a list of variables to store the accumulated gradients between batchs
            accumulated_gradients = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False)
                                     for tv in trainable_variables]

            # Define an op to reset the accumulated gradient
            self.acc_gradients_zero_op = [tv.assign(tf.zeros_like(tv)) for tv in accumulated_gradients]

            # Define an op to accumulate the gradients calculated by the current batch with
            # the accumulated gradients variable
            self.accumulate_gradients_op = [accumulated_gradients[i].assign_add(gv[0])
                                            for i, gv in enumerate(gradients)]

            # Define an op to apply the result of the accumulated gradients
            clipped_gradients, _norm = tf.clip_by_global_norm(accumulated_gradients, grad_clip)
            self.train_step_op = opt.apply_gradients([(clipped_gradients[i], gv[1]) for i, gv in enumerate(gradients)],
                                                     global_step=self.global_step)
        return learning_rate_var

    def add_tensorboard(self, session, tensorboard_dir, tb_run_name=None, timeline_enabled=False):
        """
        Add the tensorboard operations to the acoustic RNN
        This method will add ops to feed tensorboard
          self.train_summaries_op : will produce the summary for a training step
          self.test_summaries_op : will produce the summary for a test step
          self.summary_writer_op : will write the summary to disk

        Parameters
        ----------
        :param session: the tensorflow session
        :param tensorboard_dir: path to tensorboard directory
        :param tb_run_name: directory name for the tensorboard files inside tensorboard_dir, if None a default dir
                            will be created
        :param timeline_enabled: enable the output of a trace file for timeline visualization
        """
        self.tensorboard_dir = tensorboard_dir
        self.timeline_enabled = timeline_enabled

        # Define GraphKeys for TensorBoard
        graphkey_training = tf.GraphKeys()
        graphkey_test = tf.GraphKeys()

        # Learning rate
        tf.summary.scalar('Learning_rate', self.learning_rate_var, collections=[graphkey_training, graphkey_test])

        # Loss
        with tf.name_scope('Mean_loss'):
            mean_loss = tf.divide(self.accumulated_mean_loss, self.mini_batch)
            tf.summary.scalar('Training', mean_loss, collections=[graphkey_training])
            tf.summary.scalar('Test', mean_loss, collections=[graphkey_test])

        # Accuracy
        with tf.name_scope('Accuracy_-_Error_Rate'):
            mean_error_rate = tf.divide(self.accumulated_error_rate, self.mini_batch)
            tf.summary.scalar('Training', mean_error_rate, collections=[graphkey_training])
            tf.summary.scalar('Test', mean_error_rate, collections=[graphkey_test])

        self.train_summaries_op = tf.summary.merge_all(key=graphkey_training)
        self.test_summaries_op = tf.summary.merge_all(key=graphkey_test)
        if tb_run_name is None:
            run_name = datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
        else:
            run_name = tb_run_name
        self.summary_writer_op = tf.summary.FileWriter(tensorboard_dir + '/' + run_name + '/', graph=session.graph)

    def get_learning_rate(self):
        return self.learning_rate_var.eval()

    def set_learning_rate(self, sess, learning_rate):
        assign_op = self.learning_rate_var.assign(learning_rate)
        sess.run(assign_op)

    @staticmethod
    def initialize(sess):
        # Initialize variables
        sess.run(tf.global_variables_initializer())

    def save(self, session, checkpoint_dir):
        # Save the model
        checkpoint_path = os.path.join(checkpoint_dir, "acousticmodel.ckpt")
        self.saver_op.save(session, checkpoint_path, global_step=self.global_step)
        logging.info("Checkpoint saved")

    def restore(self, session, checkpoint_dir):
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        # Restore from checkpoint (will overwrite variables)
        if ckpt:
            logging.info("Reading model parameters from %s", ckpt.model_checkpoint_path)
            self.saver_op.restore(session, ckpt.model_checkpoint_path)
        else:
            logging.info("Created model with fresh parameters.")
        return

    @staticmethod
    def add_saving_op():
        """
        Define a tensorflow operation to save or restore the network

        :return: a tensorflow tf.train.Saver operation
        """
        # Define an op to save or restore the network

        # Only save needed tensors :
        #   - weight and biais from the input layer, the output layer and the LSTM
        #   - currents global_step and learning_rate
        save_list = [var for var in tf.global_variables()
                     if (var.name.find('/input_w:0') != -1) or (var.name.find('/input_b:0') != -1) or
                        (var.name.find('/output_w:0') != -1) or (var.name.find('/output_w:0') != -1) or
                        (var.name.find('global_step:0') != -1) or (var.name.find('learning_rate:0') != -1) or
                        (var.name.find('/weights:0') != -1) or (var.name.find('/biases:0') != -1)]
        if len(save_list) == 0:
            raise ValueError("Trying to define the saving operation before the RNN is built")

        saver_op = tf.train.Saver(save_list)
        return saver_op

    def get_str_labels(self, _str):
        """
        Convert a string into a label vector for the model
        The char map follow recommendations from : https://arxiv.org/pdf/1609.05935v2.pdf

        Parameters
        ----------
        _str : the string to convert into a label

        Returns
        -------
        vector of int
        """
        # add eos char
        _str += self.char_map[-1]
        # Remove spaces and set each word start with a capital letter
        next_is_upper = True
        result = []
        for i in range(len(_str)):
            if _str[i] is ' ':
                next_is_upper = True
            else:
                if next_is_upper:
                    result.append(_str[i].upper())
                    next_is_upper = False
                else:
                    result.append(_str[i])
        _str = "".join(result)
        # Convert to self.char_map indexes
        result = []
        i = 0
        while i < len(_str):
            if len(_str) - i >= 3:
                try:
                    result.append(self.char_map.index(_str[i:i+3]))
                    i += 3
                    continue
                except ValueError:
                    pass
            if len(_str) - i >= 2:
                try:
                    result.append(self.char_map.index(_str[i:i+2]))
                    i += 2
                    continue
                except ValueError:
                    pass
            try:
                result.append(self.char_map.index(_str[i:i+1]))
                i += 1
                continue
            except ValueError:
                logging.warning("Unable to process label : %s", _str)
                return []
        return result

    def get_labels_str(self, label):
        """
        Convert a vector issued from the model into a readable string

        Parameters
        ----------
        label : a vector of int containing the predicted label

        Returns
        -------
        string
        """
        # Convert int to values in self.char_map
        char_list = [self.char_map[index] for index in label if 0 <= index < len(self.char_map)]
        # Remove eos character if present
        try:
            char_list.remove(self.char_map[-1])
        except ValueError:
            pass
        # Add spaces in front of capitalized letters (except the first one) and lower every letter
        result = []
        for i in range(len(char_list)):
            if (i != 0) and (char_list[i].isupper()):
                result.append(" ")
            result.append(char_list[i].lower())
        return "".join(result)

    @staticmethod
    def calculate_wer(first_string, second_string):
        """
        Source : https://martin-thoma.com/word-error-rate-calculation/

        Calculation of WER with Levenshtein distance.

        Works only for strings up to 254 characters (uint8).
        O(nm) time ans space complexity.

        Parameters
        ----------
        first_string : string
        second_string : string

        Returns
        -------
        int

        Examples
        --------
        > calculate_wer("who is there", "is there")
        1
        > calculate_wer("who is there", "")
        3
        > calculate_wer("", "who is there")
        3
        """
        # initialisation
        r = first_string.split()
        h = second_string.split()

        d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint8)
        d = d.reshape((len(r) + 1, len(h) + 1))
        for i in range(len(r) + 1):
            for j in range(len(h) + 1):
                if i == 0:
                    d[0][j] = j
                elif j == 0:
                    d[i][0] = i

        # computation
        for i in range(1, len(r) + 1):
            for j in range(1, len(h) + 1):
                if r[i - 1] == h[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    substitution = d[i - 1][j - 1] + 1
                    insertion = d[i][j - 1] + 1
                    deletion = d[i - 1][j] + 1
                    d[i][j] = min(substitution, insertion, deletion)

        return d[len(r)][len(h)]

    @staticmethod
    def calculate_cer(first_string, second_string):
        """
        Calculation of Character Error Rate (CER).

        Works only for strings up to 65635 elements (uint16).

        Parameters
        ----------
        first_string : string
        second_string : string

        Returns
        -------
        int

        Examples
        --------
        > calculate_cer("who is there", "whois there")
        0
        > calculate_cer("who is there", "who i thre")
        2
        > calculate_cer("", "who is there")
        10
        """
        # initialisation
        r = list(first_string.replace(" ", ""))
        h = list(second_string.replace(" ", ""))

        d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint16)
        d = d.reshape((len(r) + 1, len(h) + 1))
        for i in range(len(r) + 1):
            for j in range(len(h) + 1):
                if i == 0:
                    d[0][j] = j
                elif j == 0:
                    d[i][0] = i

        # computation
        for i in range(1, len(r) + 1):
            for j in range(1, len(h) + 1):
                if r[i - 1] == h[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    substitution = d[i - 1][j - 1] + 1
                    insertion = d[i][j - 1] + 1
                    deletion = d[i - 1][j] + 1
                    d[i][j] = min(substitution, insertion, deletion)

        return d[len(r)][len(h)]

    def train_step(self, session, compute_gradients=True, run_options=None, run_metadata=None):
        """
        Returns:
        mean of ctc_loss
        """
        # Base output is to accumulate loss, error_rate and increase the mini-batchs counter
        output_feed = [self.mini_batch, self.acc_mean_loss_op, self.acc_error_rate_op, self.increase_mini_batch_op]

        if compute_gradients:
            # Add the update operation
            output_feed.append(self.accumulate_gradients_op)
            # and feed the dropout layer the keep probability values
            input_feed = {self.input_keep_prob_ph: self.input_keep_prob,
                          self.output_keep_prob_ph: self.output_keep_prob}
        else:
            # No need to apply a dropout, set the keep probability to 1.0
            input_feed = {self.input_keep_prob_ph: 1.0, self.output_keep_prob_ph: 1.0}

        # Actually run the tensorflow session
        start_time = time.time()
        logging.debug("Starting a train step")
        outputs = session.run(output_feed, input_feed, options=run_options, run_metadata=run_metadata)
        mini_batch_num = outputs[0]
        logging.debug("Step duration : %.2f", time.time() - start_time)
        return mini_batch_num

    def start_batch(self, session, is_training, run_options=None, run_metadata=None):
        output = [self.acc_error_rate_zero_op, self.acc_mean_loss_zero_op,
                  self.mini_batch_zero_op, self.rnn_state_zero_op]

        if is_training:
            output.append(self.acc_gradients_zero_op)

        session.run(output, options=run_options, run_metadata=run_metadata)
        return

    def end_batch(self, session, is_training, run_options=None, run_metadata=None):
        # Get each accumulator's value and compute the mean for the batch
        output_feed = [self.accumulated_mean_loss, self.accumulated_error_rate, self.mini_batch, self.global_step]

        # Append the train_step if needed (it will apply the gradients)
        if is_training:
            output_feed.append(self.train_step_op)

        # If a tensorboard dir is configured then run the merged_summaries operation
        if self.tensorboard_dir is not None:
            if is_training:
                output_feed.append(self.train_summaries_op)
            else:
                output_feed.append(self.test_summaries_op)

        outputs = session.run(output_feed, options=run_options, run_metadata=run_metadata)
        accumulated_loss = outputs[0]
        accumulated_error_rate = outputs[1]
        batchs_count = outputs[2]
        global_step = outputs[3]

        if self.tensorboard_dir is not None:
            summary = outputs[-1]
            self.summary_writer_op.add_summary(summary, global_step)

        mean_loss = accumulated_loss / batchs_count
        mean_error_rate = accumulated_error_rate / batchs_count
        return mean_loss, mean_error_rate, global_step

    def process_input(self, session, inputs, input_seq_lengths, run_options=None, run_metadata=None):
        """
        Returns:
          Translated text
        """
        input_feed = {self.input_keep_prob_ph: 1.0, self.output_keep_prob_ph: 1.0, self.inputs_ph: np.array(inputs),
                      self.input_seq_lengths_ph: np.array(input_seq_lengths)}
        output_feed = [self.prediction]
        outputs = session.run(output_feed, input_feed, options=run_options, run_metadata=run_metadata)
        predictions = session.run(tf.sparse_tensor_to_dense(outputs[0], default_value=len(self.char_map),
                                                            validate_indices=True),
                                  options=run_options, run_metadata=run_metadata)
        transcribed_text = [self.get_labels_str(prediction) for prediction in predictions]
        return transcribed_text

    def evaluate_full(self, sess, eval_dataset, audio_processor, run_options=None, run_metadata=None):
        wer_list = []
        cer_list = []
        file_number = 0
        input_feat_vecs = []
        input_feat_vec_lengths = []
        labels = []
        for file, label, _ in eval_dataset:
            feat_vec, feat_vec_length = audio_processor.process_audio_file(file)
            file_number += 1
            label_data_length = len(label)
            if (label_data_length > self.max_target_seq_length) or\
               (feat_vec_length > self.max_input_seq_length):
                logging.warning("Warning - sample too long : %s (input : %d / text : %s)",
                                file, feat_vec_length, label_data_length)
            else:
                logging.debug("Processed file %d / %d", file_number, len(eval_dataset))
                input_feat_vecs.append(feat_vec)
                input_feat_vec_lengths.append(feat_vec_length)
                labels.append(label)

            # If we reached the last file then pad the lists to obtain a full batch
            if file_number == len(eval_dataset):
                for i in range(self.batch_size - len(input_feat_vecs)):
                    input_feat_vecs.append(np.zeros([self.max_input_seq_length,
                                                     audio_processor.feature_size]))
                    input_feat_vec_lengths.append(0)
                    labels.append("")

            if len(input_feat_vecs) == self.batch_size:
                # Run the batch
                logging.debug("Running a batch")
                input_feat_vecs = np.swapaxes(input_feat_vecs, 0, 1)
                transcribed_texts = self.process_input(sess, input_feat_vecs, input_feat_vec_lengths,
                                                       run_options=run_options, run_metadata=run_metadata)
                for index, transcribed_text in enumerate(transcribed_texts):
                    true_label = labels[index]
                    if len(true_label) > 0:
                        nb_words = len(true_label.split())
                        nb_chars = len(true_label.replace(" ", ""))
                        wer_list.append(self.calculate_wer(transcribed_text, true_label) / float(nb_words))
                        cer_list.append(self.calculate_cer(transcribed_text, true_label) / float(nb_chars))
                # Reset the lists
                input_feat_vecs = []
                input_feat_vec_lengths = []
                labels = []

        wer = (sum(wer_list) * 100) / float(len(wer_list))
        cer = (sum(cer_list) * 100) / float(len(cer_list))
        return wer, cer

    def evaluate_basic(self, sess, eval_dataset, audio_processor, run_options=None, run_metadata=None):
        start_time = time.time()
        # Create a thread to load data
        _ = self.enqueue_data(sess, audio_processor, eval_dataset, run_forever=False,
                              run_options=run_options, run_metadata=run_metadata)

        # Main evaluation loop
        logging.info("Start evaluating...")

        # Start a new batch
        self.start_batch(sess, False, run_options=run_options, run_metadata=run_metadata)

        try:
            while True:
                self.train_step(sess, False, run_options=run_options, run_metadata=run_metadata)
        except tf.errors.InvalidArgumentError:
            logging.debug("Queue empty, exiting evaluation step")

        # Close the batch
        mean_loss, mean_error_rate, current_step = self.end_batch(sess, False, run_options=run_options,
                                                                  run_metadata=run_metadata)
        logging.info("Evaluation at step %d : loss %.5f - error_rate %.5f - duration %.2f",
                     current_step, mean_loss, mean_error_rate, time.time() - start_time)

        return mean_loss, mean_error_rate, current_step

    def enqueue_data(self, sess, audio_processor, dataset, run_forever=False, run_options=None, run_metadata=None):
        # Create a coordinator for the queue if there isn't or reset the existing one
        if self.coord is None:
            self.coord = tf.train.Coordinator()
        else:
            self.coord.clear_stop()

        # Build a thread to process input data into the queue
        thread_local_data = threading.local()
        thread = threading.Thread(name="thread_enqueue", target=self._enqueue_data_thread,
                                  args=(self.coord, self.thread_lock, sess, audio_processor, thread_local_data,
                                        dataset, run_forever, run_options, run_metadata))
        thread.start()
        return thread

    def _enqueue_data_thread(self, coord, lock, sess, audio_processor, t_local_data, dataset,
                             run_forever=False, run_options=None, run_metadata=None):
        # Make a local copy of the dataset
        t_local_data.dataset = dataset[:]
        t_local_data.current_pos = 0

        while not coord.should_stop():
            # Start over if end is reached
            if t_local_data.current_pos >= len(t_local_data.dataset):
                if run_forever:
                    logging.debug("An epoch have been reached, starting a new one")
                    t_local_data.current_pos = 0
                else:
                    logging.debug("An epoch have been reached, sending end signal")
                    # Sending a full batch of end signal items in the queue to detect ending
                    # A full batch is needed to be sure that the dequeue op can be made
                    for i in range(self.batch_size):
                        sess.run(self.enqueue_op, feed_dict={self.queue_input_ph: np.zeros([1, self.input_dim]),
                                                             self.queue_input_length_ph: 0,
                                                             self.queue_label_ph: [0],
                                                             self.queue_end_signal_ph: [True]},
                                 options=run_options, run_metadata=run_metadata)
                    break

            # Take an item in the list and update position
            [t_local_data.file, t_local_data.text, _] = t_local_data.dataset[t_local_data.current_pos]
            t_local_data.current_pos += 1

            # Calculate MFCC
            lock.acquire()
            try:
                t_local_data.mfcc_data, t_local_data.original_mfcc_length =\
                    audio_processor.process_audio_file(t_local_data.file)
                logging.debug("File %s processed, resulting a array of shape %s - original signal size is %d",
                              t_local_data.file, t_local_data.mfcc_data.shape, t_local_data.original_mfcc_length)
            finally:
                lock.release()

            # Convert string to numbers
            t_local_data.label_data = self.get_str_labels(t_local_data.text)
            if len(t_local_data.label_data) == 0:
                # Incorrect label
                logging.warning("Incorrect label for %s (%s)", t_local_data.file, t_local_data.text)
                continue
            # Check sizes and pad if needed
            t_local_data.label_data_length = len(t_local_data.label_data)
            if (t_local_data.label_data_length > self.max_target_seq_length) or\
               (t_local_data.original_mfcc_length > self.max_input_seq_length):
                # Either input or output vector is too long
                logging.warning("Warning - sample too long : %s (input : %d / text : %s)",
                                t_local_data.file, t_local_data.original_mfcc_length, t_local_data.label_data_length)
                continue
            elif t_local_data.label_data_length < self.max_target_seq_length:
                # Label need padding
                t_local_data.label_data += [0] * (self.max_target_seq_length - len(t_local_data.label_data))

            try:
                sess.run(self.enqueue_op, feed_dict={self.queue_input_ph: t_local_data.mfcc_data,
                                                     self.queue_input_length_ph: t_local_data.original_mfcc_length,
                                                     self.queue_label_ph: t_local_data.label_data,
                                                     self.queue_end_signal_ph: [False]},
                         options=run_options, run_metadata=run_metadata)
            except tf.errors.CancelledError:
                # The queue have been cancelled so we should stop
                logging.debug("Queue has been closed, exiting input thread")
                break

    def _write_timeline(self, run_metadata, inter_time, action=""):
        logging.debug("--- Action %s duration : %.4f", action, time.time() - inter_time)

        if self.tensorboard_dir is None:
            logging.warning("Could not write timeline, a tensorboard_dir is required in config file")
            return

        # Create the Timeline object, and write it to a json
        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        logging.info('Writing to timeline-' + action + '.ctf.json')
        with open(self.tensorboard_dir + '/' + 'timeline-' + action + '.ctf.json', 'w') as trace_file:
            trace_file.write(trace.generate_chrome_trace_format())
        return time.time()

    def run_train_step(self, sess, mini_batch_size, run_options=None, run_metadata=None):
        """
        Run a single train step 

        Parameters
        ----------
        :param sess: a tensorflow session
        :param mini_batch_size: the number of batchs to run before applying the gradients
        :param run_options: options parameter for the sess.run calls
        :param run_metadata: run_metadata parameter for the sess.run calls
        :returns float mean_loss: mean loss for the train batch run
        :returns float mean_error_rate: mean error rate for the train batch run
        :returns int current_step: new value of the step counter at the end of this batch
        :returns bool queueing_finished: `True` if the queue was emptied during the batch
        """
        start_time = inter_time = time.time()
        queueing_finished = False

        # Start a new batch
        self.start_batch(sess, True, run_options=run_options, run_metadata=run_metadata)
        if self.timeline_enabled:
            inter_time = self._write_timeline(run_metadata, inter_time, "start_batch")

        # Run multiple mini-batchs inside the train step
        mini_batch_num = 0
        try:
            for i in range(mini_batch_size):
                # Run a step on a batch and keep the loss
                mini_batch_num = self.train_step(sess, True, run_options=run_options, run_metadata=run_metadata)
                if self.timeline_enabled:
                    inter_time = self._write_timeline(run_metadata, inter_time, "step-" + str(i))
        except tf.errors.InvalidArgumentError:
            logging.debug("Queue empty, exiting train step")
            queueing_finished = True

        # Close the batch if at least a mini-batch was completed
        if mini_batch_num > 0:
            mean_loss, mean_error_rate, current_step = self.end_batch(sess, True, run_options=run_options,
                                                                      run_metadata=run_metadata)
            if self.timeline_enabled:
                _ = self._write_timeline(run_metadata, inter_time, "end_batch")

            # Step result
            logging.info("Batch %d : loss %.5f - error_rate %.5f - duration %.2f",
                         current_step, mean_loss, mean_error_rate, time.time() - start_time)

            return mean_loss, mean_error_rate, current_step, queueing_finished
        else:
            return 0.0, 0.0, self.global_step.eval(), queueing_finished

    def fit(self, sess, audio_processor, train_set, mini_batch_size, max_steps=None,
            run_options=None, run_metadata=None):
        """
        Fit the model to the given train_set.
        
        The fitting process will proceed until the end of the train_set or max_steps is reached
        
        Parameters
        ----------
        :param sess: a tensorflow session
        :param audio_processor: an AudioProcessor object to convert audio files
        :param train_set: the train_set to fit
        :param mini_batch_size: the number of batchs to run before applying the gradients
        :param max_steps: max number of steps to run
        :param run_options: options parameter for the sess.run calls
        :param run_metadata: run_metadata parameter for the sess.run calls
        :return: step number at the end of the training
        """
        # Create a thread to load data
        thread = self.enqueue_data(sess, audio_processor, train_set, run_forever=False,
                                   run_options=run_options, run_metadata=run_metadata)

        # Main training loop
        logging.info("Start fit with %d files", len(train_set))
        queueing_finished = False
        current_step = 0
        while not queueing_finished:
            if self.coord.should_stop():
                break

            _, _, current_step, queueing_finished = self.run_train_step(sess, mini_batch_size, run_options=run_options,
                                                                        run_metadata=run_metadata)

            if (max_steps is not None) and (current_step >= max_steps):
                # Maximum allowed reached, exiting the training process
                logging.debug("Max steps reached, exiting.")
                break

        if not queueing_finished:
            # Ask the threads to stop.
            logging.debug("Asking for threads to stop")
            self.coord.request_stop()
            # Empty the queue
            sess.run(self.queue.dequeue_up_to(self.batch_size * 2), options=run_options, run_metadata=run_metadata)
            # And wait for them to actually stop
            self.coord.join([thread], stop_grace_period_secs=10)

        return current_step
