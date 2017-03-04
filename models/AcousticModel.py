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
    def __init__(self, session, num_layers, hidden_size, input_keep_prob, output_keep_prob,
                 batch_size, learning_rate, lr_decay_factor, grad_clip,
                 max_input_seq_length, max_target_seq_length, input_dim, normalization,
                 forward_only=False, tensorboard_dir=None, tb_run_name=None,
                 timeline_enabled=False, language='english'):
        """
        Acoustic rnn model, using ctc loss with lstm cells
        Inputs:
        session - tensorflow session
        num_layers - number of lstm layers
        hidden_size - size of hidden layers
        input_keep_prob - probability of keeping input signal for a cell during training
        output_keep_prob - probability of keeping output signal from a cell during training
        batch_size - number of training examples fed at once
        learning_rate - learning rate parameter fed to optimizer
        lr_decay_factor - decay factor of the learning rate
        grad_clip - max gradient size (prevent exploding gradients)
        max_input_seq_length - maximum length of input vector sequence
        max_target_seq_length - maximum length of ouput vector sequence
        input_dim - dimension of input vector
        normalization - boolean indicating whether or not to normalize data in a input batch
        forward_only - whether to build back prop nodes or not
        tensorboard_dir - path to tensorboard file (None if not activated)
        tb_run_name - directory name for the tensorboard files (inside tensorboard_dir, None mean no sub-directory)
        timeline_enabled - enable the output of a trace file for timeline visualization
        language - the language of the speech
        """
        # Initialize thread management
        self.lock = threading.Lock()

        # Set language
        if language == 'english':
            self.char_map = ENGLISH_CHAR_MAP
        else:
            raise ValueError("Invalid parameter 'language' for method '__init__'")
        num_labels = len(self.char_map)

        # Declare object variables used as inputs for the RNN
        self.inputs = None
        self.input_seq_lengths = None
        self.input_keep_prob_ph = None
        self.output_keep_prob_ph = None
        # Declare object variable used as the RNN internal state
        self.hidden_state = None
        # Declare object variable used as the RNN output
        self.prediction = None
        # Build the RNN
        logits = self.build_base_rnn(hidden_size, num_layers, num_labels, input_dim, batch_size,
                                     max_input_seq_length, forward_only, normalization)

        # Add the train part to the network if needed
        if not forward_only:
            # Declare object variables used as truth label for training the RNN
            self.sparse_labels = None
            # Declare object variables used as the output of the training operation
            self.mean_loss = None
            # Declare object variables used as a training parameter
            self.learning_rate = None
            self.global_step = None
            # Declare ops for training
            self.learning_rate_decay_op = None
            self.acc_gradients_zero_op = None
            self.accumulate_gradients_op = None
            self.train_step_op = None
            # Add the train part to the network
            self.build_train_rnn(logits, grad_clip, learning_rate, lr_decay_factor)

        # Store model parameters
        self.input_keep_prob = input_keep_prob
        self.output_keep_prob = output_keep_prob
        self.batch_size = batch_size
        self.max_input_seq_length = max_input_seq_length
        self.max_target_seq_length = max_target_seq_length
        self.tensorboard_dir = tensorboard_dir
        self.timeline_enabled = timeline_enabled
        self.input_dim = input_dim

        # TensorBoard init
        if (self.tensorboard_dir is not None) and (not forward_only):
            self.train_summaries_op = None
            self.test_summaries_op = None
            self.summary_writer_op = None
            self.tensorboard_rnn(session, tensorboard_dir, tb_run_name)

        # Finally define an op to save or restore the network
        # Only save needed tensors :
        #   - weight and biais from the input layer, the output layer and the LSTM
        #   - currents global_step and leaning_rate
        save_list = [var for var in tf.global_variables()
                     if (var.name.find('/input_w:0') != -1) or (var.name.find('/input_b:0') != -1) or
                        (var.name.find('/output_w:0') != -1) or (var.name.find('/output_w:0') != -1) or
                        (var.name.find('global_step:0') != -1) or (var.name.find('learning_rate:0') != -1) or
                        (var.name.find('/weights:0') != -1) or (var.name.find('/biases:0') != -1)]

        for var in tf.global_variables():
            logging.debug("TF variable : %s - %s", var.name, var)

        self.saver = tf.train.Saver(save_list)

    def build_base_rnn(self, hidden_size, num_layers, num_labels, input_dim, batch_size,
                       max_input_seq_length, forward_only, normalization):
        """
        Build the Acoustic RNN

        Parameters
        ----------
        hidden_size : size of hidden layers (number of lstm cells on each hidden layer)
        num_layers : number of lstm layers
        num_labels : number of possible labels on output
        input_dim : dimension of an input vector
        batch_size : number of training examples fed at once
        max_input_seq_length : maximum length of input vector sequence
        forward_only : whether to build back prop nodes or not
        normalization : boolean indicating whether or not to normalize data in a input batch

        Returns
        -------
        the logits (each char probability for each timestep of the input, for each item of the batch)
        """
        # Define the RNN input placeholder
        self.inputs = tf.placeholder(tf.float32, shape=[max_input_seq_length, None, input_dim], name="inputs")

        # Define a placeholder for the effective length of each input vector
        # We could take an int16 for less memory consumption but CTC need an int32
        self.input_seq_lengths = tf.placeholder(tf.int32, shape=[None], name="input_seq_lengths")

        # Define cells of acoustic model
        with tf.name_scope('LSTM'):
            cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)

        # Define a dropout layer (used only when training)
        with tf.name_scope('dropout'):
            # Create placeholders, used to override values when running on the test set
            self.input_keep_prob_ph = tf.placeholder(tf.float32)
            self.output_keep_prob_ph = tf.placeholder(tf.float32)
            if not forward_only:
                # If we are in training then add a dropoutWrapper to the cells
                cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.input_keep_prob_ph,
                                                     output_keep_prob=self.output_keep_prob_ph)

        with tf.name_scope('LSTM'):
            if num_layers > 1:
                cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

        # Build the input layer between input and the RNN
        with tf.name_scope('Input_Layer'):
            w_i = tf.Variable(tf.truncated_normal([input_dim, hidden_size], stddev=np.sqrt(2.0 / (2 * hidden_size))),
                              name="input_w")
            b_i = tf.Variable(tf.zeros([hidden_size]), name="input_b")

        # Apply the input layer to the network input to produce the input for the rnn part of the network
        rnn_inputs = [tf.matmul(tf.squeeze(i, axis=[0]), w_i) + b_i
                      for i in tf.split(axis=0, num_or_size_splits=max_input_seq_length, value=self.inputs)]
        # Switch from a list to a tensor
        rnn_inputs = tf.stack(rnn_inputs)

        # If we are in training then add a batch normalization layer to the model
        with tf.name_scope('Normalization'):
            if normalization and not forward_only:
                epsilon = 1e-3
                # Note : the tensor is [time, batch_size, input vector] so we go against dim 1
                batch_mean, batch_var = tf.nn.moments(rnn_inputs, [1], shift=None, name="moments", keep_dims=True)
                rnn_inputs = tf.nn.batch_normalization(rnn_inputs, batch_mean, batch_var, None, None,
                                                       epsilon, name="batch_norm")

        # Set the RNN initial state to 0s
        init_state = cell.zero_state(batch_size, tf.float32)

        # Build the RNN
        with tf.name_scope('LSTM'):
            rnn_output, self.hidden_state = tf.nn.dynamic_rnn(cell, rnn_inputs, sequence_length=self.input_seq_lengths,
                                                              initial_state=init_state, time_major=True)

        # Build the output layer between the RNN and the char_map
        with tf.name_scope('Output_layer'):
            w_o = tf.Variable(tf.truncated_normal([hidden_size, num_labels], stddev=np.sqrt(2.0 / (2 * num_labels))),
                              name="output_w")
            b_o = tf.Variable(tf.zeros([num_labels]), name="output_b")

        # Compute the logits (each char probability for each timestep of the input, for each item of the batch)
        logits = tf.stack([tf.matmul(tf.squeeze(i, axis=[0]), w_o) + b_o
                          for i in tf.split(axis=0, num_or_size_splits=max_input_seq_length, value=rnn_output)])

        # Compute the prediction which is the best "path" of probabilities for each item of the batch
        decoded, _log_prob = tf.nn.ctc_beam_search_decoder(logits, self.input_seq_lengths)
        # Set the RNN result to the best path found
        self.prediction = tf.to_int32(decoded[0])

        return logits

    def build_train_rnn(self, logits, grad_clip, learning_rate, lr_decay_factor):
        """
        Build the training add-on of the Acoustic RNN
        This add-on offer ops that can be used to train the network
          self.learning_rate_decay_op : will decay the learning rate
          self.acc_gradients_zero_op : will reset the gradients
          self.accumulate_gradients_op : will compute the gradients and accumulate them over multiple batchs
          self.train_step_op : will clip the accumulated gradients and apply them on the RNN

        Parameters
        ----------
        logits : the output of the RNN before the beam search
        grad_clip : max gradient size (prevent exploding gradients)
        learning_rate : learning rate parameter fed to optimizer
        lr_decay_factor : decay factor of the learning rate

        Returns
        -------
        None
        """
        # Define a variable to keep track of the learning process step
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # Define the sparse tensor used as truth label for training the RNN
        self.sparse_labels = tf.sparse_placeholder(tf.int32)

        # Define the variable for the learning rate
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, name='learning_rate')
        # Define an op to decrease the learning rate
        self.learning_rate_decay_op = self.learning_rate.assign(tf.multiply(self.learning_rate, lr_decay_factor))

        # Compute the CTC loss between the logits and the truth for each item of the batch
        with tf.name_scope('CTC'):
            ctc_loss = tf.nn.ctc_loss(self.sparse_labels, logits, self.input_seq_lengths)

            # Compute the mean loss of the batch (only used to check on progression in learning)
            # The loss is averaged accross the batch but before we take into account the real size of the label
            self.mean_loss = tf.reduce_mean(tf.truediv(ctc_loss, tf.to_float(self.input_seq_lengths)))

        # Compute the gradients
        trainable_variables = tf.trainable_variables()
        with tf.name_scope('Gradients'):
            opt = tf.train.AdamOptimizer(self.learning_rate)
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
        return

    def tensorboard_rnn(self, session, tensorboard_dir, tb_run_name):
        """
        Build the tensorboard operations of the Acoustic RNN
        This method will add ops to feed tensorboard
          self.train_summaries_op : will produce the summary for a training step
          self.test_summaries_op : will produce the summary for a test step
          self.summary_writer_op : will write the summary to disk

        Parameters
        ----------
        session : the tensorflow session
        tensorboard_dir : path to tensorboard file (None if not activated)
        tb_run_name : directory name for the tensorboard files (inside tensorboard_dir, None mean no sub-directory)

        Returns
        -------
        None
        """
        # Define GraphKeys for TensorBoard
        graphkey_training = tf.GraphKeys()
        graphkey_test = tf.GraphKeys()

        # Learning rate
        tf.summary.scalar('Learning_rate', self.learning_rate, collections=[graphkey_training, graphkey_test])

        # Loss
        with tf.name_scope('Mean_loss'):
            tf.summary.scalar('Training', self.mean_loss, collections=[graphkey_training])
            tf.summary.scalar('Test', self.mean_loss, collections=[graphkey_test])

        # Accuracy
        with tf.name_scope('Accuracy_-_Error_Rate'):
            error_rate = tf.reduce_mean(tf.edit_distance(self.prediction, self.sparse_labels, normalize=True))
            tf.summary.scalar('Training', error_rate, collections=[graphkey_training])
            tf.summary.scalar('Test', error_rate, collections=[graphkey_test])

        self.train_summaries_op = tf.summary.merge_all(key=graphkey_training)
        self.test_summaries_op = tf.summary.merge_all(key=graphkey_test)
        if tb_run_name is None:
            run_name = datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
        else:
            run_name = tb_run_name
        self.summary_writer_op = tf.summary.FileWriter(tensorboard_dir + '/' + run_name + '/', graph=session.graph)

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

    def get_num_batches(self, dataset):
        return len(dataset) // self.batch_size

    def step(self, session, input_feat_vecs, mfcc_lengths_batch,
             label_values_batch, label_indices_batch, is_training):
        """
        Returns:
        mean of ctc_loss
        """
        input_feed = {self.inputs: input_feat_vecs, self.input_seq_lengths: mfcc_lengths_batch,
                      self.sparse_labels: tf.SparseTensorValue(label_indices_batch, label_values_batch,
                                                               [self.batch_size, self.max_target_seq_length])}
        # Base output is ctc_loss and mean_loss
        output_feed = [self.mean_loss]

        # If a tensorboard dir is configured then add a merged_summaries operation
        run_options = run_metadata = None
        if self.tensorboard_dir is not None:
            if is_training:
                output_feed.append(self.train_summaries_op)
            else:
                output_feed.append(self.test_summaries_op)

            # Add timeline data generation options if needed
            if self.timeline_enabled is True:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

        if is_training:
            # If in training then add the update operation
            output_feed.append(self.accumulate_gradients_op)
            # and feed the dropout layer the keep probability values
            input_feed[self.input_keep_prob_ph] = self.input_keep_prob
            input_feed[self.output_keep_prob_ph] = self.output_keep_prob
        else:
            # If not in training then no need to apply a dropout, set the keep probability to 1.0
            input_feed[self.input_keep_prob_ph] = 1.0
            input_feed[self.output_keep_prob_ph] = 1.0

        # Actually run the tensorflow session
        start_time = time.time()
        outputs = session.run(output_feed, input_feed, options=run_options, run_metadata=run_metadata)
        logging.debug("Step duration : %.2f", time.time() - start_time)

        # If a tensorboard dir is configured then generate the summary for this operation
        if self.tensorboard_dir is not None:
            self.summary_writer_op.add_summary(outputs[1], self.global_step.eval())

            # ...and produce the timeline if needed
            if self.timeline_enabled is True:
                # Create the Timeline object, and write it to a json
                from tensorflow.python.client import timeline
                tl = timeline.Timeline(run_metadata.step_stats)
                ctf = tl.generate_chrome_trace_format()
                with open(self.tensorboard_dir + '/' + 'timeline.json', 'w') as f:
                    logging.info("Writing to timeline.json")
                    f.write(ctf)

        return outputs[0]

    def process_input(self, session, inputs, input_seq_lengths):
        """
        Returns:
          Translated text
        """
        input_feed = {self.input_keep_prob_ph: 1.0, self.output_keep_prob_ph: 1.0, self.inputs.name: np.array(inputs),
                      self.input_seq_lengths.name: np.array(input_seq_lengths)}
        output_feed = [self.prediction]
        outputs = session.run(output_feed, input_feed)
        predictions = session.run(tf.sparse_tensor_to_dense(outputs[0], default_value=len(self.char_map),
                                                            validate_indices=True))
        transcribed_text = [self.get_labels_str(prediction) for prediction in predictions]
        return transcribed_text

    def run_checkpoint(self, sess, checkpoint_dir, num_test_batches, dequeue_op):
        # Save the model
        checkpoint_path = os.path.join(checkpoint_dir, "acousticmodel.ckpt")
        self.saver.save(sess, checkpoint_path, global_step=self.global_step)

        # Run a test set against the current model
        mean_loss = 0.0
        if num_test_batches > 0:
            logging.info("Test set - Will proceed to %d iterations", num_test_batches)
            for i in range(num_test_batches):
                logging.debug("On %dth iteration of the test set", i)
                input_feat_vecs, mfcc_lengths_batch, label_values_batch, label_indices_batch = \
                    self.dequeue_data(sess, dequeue_op)

                step_loss = self.step(sess, input_feat_vecs, mfcc_lengths_batch, label_values_batch,
                                      label_indices_batch, False)
                mean_loss += step_loss / num_test_batches
            logging.info("Finished test set - resulting loss is %.2f", mean_loss)
        return mean_loss

    def enqueue_data(self, coord, sess, audio_processor, t_local_data, enqueue_op, dataset,
                     mfcc_input, mfcc_input_length, label, label_length, start_from=0):
        # Make a local copy of the dataset and set the reading index
        t_local_data.dataset = dataset[:]
        t_local_data.current_pos = start_from

        while not coord.should_stop():
            if t_local_data.current_pos >= len(t_local_data.dataset):
                t_local_data.current_pos = 0

            # Take an item in the list and increase position
            [t_local_data.file, t_local_data.text, _] = t_local_data.dataset[t_local_data.current_pos]

            # Calculate MFCC
            self.lock.acquire()
            try:
                t_local_data.mfcc_data, t_local_data.original_mfcc_length =\
                    audio_processor.process_audio_file(t_local_data.file)
                logging.debug("File %s processed, resulting a array of shape %s - original signal size is %d",
                              t_local_data.file, t_local_data.mfcc_data.shape, t_local_data.original_mfcc_length)
            finally:
                self.lock.release()

            # Convert string to numbers
            t_local_data.label_data = self.get_str_labels(t_local_data.text)
            if len(t_local_data.label_data) == 0:
                # Incorrect label
                logging.warning("Incorrect label for %s (%s)", t_local_data.file, t_local_data.text)
                # Remove the file from the list
                t_local_data.dataset.pop(t_local_data.current_pos)
                continue
            # Check sizes and pad if needed
            t_local_data.label_data_length = len(t_local_data.label_data)
            if (t_local_data.label_data_length > self.max_target_seq_length) or\
                    (t_local_data.original_mfcc_length > self.max_input_seq_length):
                # If either input or output vector is too long we shouldn't take this sample
                logging.warning("Warning - sample too long : %s (input : %d / text : %s)",
                                t_local_data.file, t_local_data.original_mfcc_length, t_local_data.label_data_length)
                # Remove the file from the list
                t_local_data.dataset.pop(t_local_data.current_pos)
                continue
            elif t_local_data.label_data_length < self.max_target_seq_length:
                # Label need padding
                t_local_data.label_data += [0] * (self.max_target_seq_length - len(t_local_data.label_data))

            try:
                sess.run(enqueue_op, feed_dict={mfcc_input: t_local_data.mfcc_data,
                                                mfcc_input_length: t_local_data.original_mfcc_length,
                                                label: t_local_data.label_data,
                                                label_length: t_local_data.label_data_length})
            except Exception as e:
                # The queue may have been cancelled so we should stop
                coord.request_stop(e)
                break

            # Go to next file on the list
            t_local_data.current_pos += 1

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

    def create_queue(self, sess, audio_processor, coord, dataset, queue_type):
        start_from = 0
        capacity = min(self.batch_size * 10, len(dataset))
        if queue_type == 'train':
            # Shuffle queue for the train set, but we don't shuffle too much in order to keep the benefit from
            # having homogeneous sizes in a given batch (files are ordered by size ascending)
            min_after_dequeue = min(self.batch_size * 7, len(dataset) - self.batch_size)
            queue = tf.RandomShuffleQueue(capacity, min_after_dequeue, [tf.int32, tf.int32, tf.int32, tf.int32],
                                          shapes=[[self.max_input_seq_length, self.input_dim], [],
                                                  [self.max_target_seq_length], []])
            # Calculate approximate position for learning batch, allow to keep consistency between multiple iterations
            # of the same training job (will default to 0 if it is the first launch because global_step will be 0)
            start_from = self.global_step.eval() * self.batch_size
            if start_from != 0:
                logging.info("Start training from file number : %d", start_from)
        elif queue_type == 'test':
            # Simple FIFO queue for the test set because we don't care to test always in the same order
            queue = tf.FIFOQueue(capacity, [tf.int32, tf.int32, tf.int32, tf.int32],
                                 shapes=[[self.max_input_seq_length, self.input_dim], [],
                                         [self.max_target_seq_length], []])
        else:
            raise ValueError("Invalid parameter 'queue_type' for method 'create_queue'")

        # Define the enqueue operation for data
        mfcc_input = tf.placeholder(tf.int32, shape=[self.max_input_seq_length, self.input_dim],
                                    name=queue_type + "_mfcc_input")
        mfcc_input_length = tf.placeholder(tf.int32, shape=[],
                                           name=queue_type + "_mfcc_input_length")
        label = tf.placeholder(tf.int32, shape=[self.max_target_seq_length],
                               name=queue_type + "_label")
        label_length = tf.placeholder(tf.int32, shape=[],
                                      name=queue_type + "_label_length")
        enqueue_op = queue.enqueue([mfcc_input, mfcc_input_length, label, label_length])
        dequeue_op = queue.dequeue_many(self.batch_size)
        thread_local_data = threading.local()
        thread = threading.Thread(name=queue_type + "_enqueue", target=self.enqueue_data,
                                  args=(coord, sess, audio_processor, thread_local_data, enqueue_op, dataset,
                                        mfcc_input, mfcc_input_length, label, label_length,
                                        start_from))
        return thread, dequeue_op

    def train(self, sess, audio_processor, test_set, train_set, steps_per_checkpoint,
              checkpoint_dir, nb_iterations=1, max_epoch=None):
        # Create a queue for each dataset and a coordinator
        coord = tf.train.Coordinator()
        train_thread, train_dequeue_op = self.create_queue(sess, audio_processor, coord, train_set, 'train')
        test_thread, test_dequeue_op = self.create_queue(sess, audio_processor, coord, test_set, 'test')
        threads = [train_thread, test_thread]
        for t in threads:
            t.start()

        previous_best_loss = None
        no_improvement_since = 0
        mean_step_time, mean_loss = 0.0, 0.0
        current_step = 1

        # Main training loop
        while True:
            if coord.should_stop():
                break

            step_loss = 0.0
            start_time = time.time()

            # Set accumulated gradients to zeros
            sess.run(self.acc_gradients_zero_op)

            # Run multiple batchs
            for i in range(nb_iterations):
                input_feat_vecs, mfcc_lengths_batch, label_values_batch, label_indices_batch =\
                    self.dequeue_data(sess, train_dequeue_op)

                # Run a step on a batch and keep the loss
                step_loss += self.step(sess, input_feat_vecs, mfcc_lengths_batch, label_values_batch,
                                       label_indices_batch, True)
            # Apply the gradients
            sess.run(self.train_step_op)

            # Step result
            logging.info("Step %d with loss %.4f (duration : %.2f)", current_step, step_loss / nb_iterations,
                         time.time() - start_time)
            mean_step_time += (time.time() - start_time) / steps_per_checkpoint
            mean_loss += step_loss / nb_iterations / steps_per_checkpoint

            # Check if we are at a checkpoint
            if current_step % steps_per_checkpoint == 0:
                logging.info("Global step %d / learning rate %.4f / step-time %.2f / loss %.2f",
                             self.global_step.eval(), self.learning_rate.eval(), mean_step_time, mean_loss)
                num_test_batches = self.get_num_batches(test_set)
                chkpt_loss = self.run_checkpoint(sess, checkpoint_dir, num_test_batches, test_dequeue_op)
                mean_step_time, mean_loss = 0.0, 0.0

                if num_test_batches > 0:
                    # Decrease learning rate if the model is not improving. The model is not improving if the loss
                    # does not get better after two iterations. This way it allow for a temporary degradation but it
                    # must improve, otherwise it means that the model is probably oscillating
                    if (previous_best_loss is not None) and (chkpt_loss >= previous_best_loss):
                        no_improvement_since += 1
                        if no_improvement_since == 2:
                            sess.run(self.learning_rate_decay_op)
                            no_improvement_since = 0
                            previous_best_loss = chkpt_loss
                            if self.learning_rate.eval() < 1e-7:
                                # End learning process
                                break
                    else:
                        no_improvement_since = 0
                        previous_best_loss = chkpt_loss

            current_step += 1
            if (max_epoch is not None) and (current_step > max_epoch):
                # We have reached the maximum allowed, we should exit at the end of this run
                break

        # Stop the queues
        sess.run(train_dequeue_op.close(cancel_pending_enqueues=True))
        sess.run(test_dequeue_op.close(cancel_pending_enqueues=True))
        # Ask the threads to stop.
        coord.request_stop()
        # And wait for them to actually do it.
        coord.join(threads)
