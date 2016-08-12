'''
Based on the paper:

http://arxiv.org/pdf/1601.06581v2.pdf

This model is:

acoustic RNN trained with ctc loss
'''

import tensorflow as tf
try:
    from tensorflow.models.rnn import seq2seq, rnn_cell, rnn
except:
    from tensorflow.python.ops import seq2seq, rnn_cell, rnn
import tensorflow.contrib.ctc as ctc
import util.audioprocessor as audioprocessor
import numpy as np


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
        grad_clip - max gradient size (prevent exploding gradients)
        max_seq_length - maximum length of input vector sequence
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
        # graph inputs
        self.inputs = tf.placeholder(tf.float32,
                                     shape=[self.max_input_seq_length, None, input_dim],
                                     name="inputs")
        self.input_seq_lengths = tf.placeholder(tf.int32,
                                                shape=[None],
                                                name="input_seq_lengths")
        self.target_seq_lengths = tf.placeholder(tf.int32,
                                                 shape=[None],
                                                 name="target_seq_lengths")
        # graph sparse tensor inputs
        self.target_indices = tf.placeholder(tf.int64,
                                             shape=[None, 2],
                                             name="target_indices")
        self.target_vals = tf.placeholder(tf.int32,
                                          shape=[None],
                                          name="target_vals")

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
        inputs = [tf.matmul(tf.squeeze(i), w_i) + b_i for i in tf.split(0, self.max_input_seq_length, self.inputs)]

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
        self.logits = [tf.matmul(tf.squeeze(i), w_o) + b_o for i in tf.split(0, self.max_input_seq_length, rnn_output)]
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

        if not forward_only:
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
        dataset - tuples of (numpy file, transcribed_text)
        Returns:
        input_feat_vecs, input_feat_vec_lengths, target_lengths,
            target_labels, target_indices
        '''
        already_processed = self.batch_size * batch_pointer
        num_data_points = min(self.batch_size, len(dataset[already_processed:]))
        to_process = dataset[already_processed:already_processed + num_data_points]
        input_feat_vecs = []
        input_feat_vec_lengths = []
        target_lengths = []
        target_labels = []
        target_indices = []

        batch_counter = 0
        for file_text in to_process:
            feat_vec, feat_vec_length = self.audio_processor.processFLACAudio(file_text[0])
            labels = self.getStrLabels(file_text[1])

            if len(feat_vec) > self.max_input_seq_length:
                feat_vec = feat_vec[:self.max_input_seq_length]
                feat_vec_length = self.max_input_seq_length
            input_feat_vecs.append(feat_vec)
            assert feat_vec_length <= self.max_input_seq_length, "{0} not less than {1}".format(feat_vec_length,
                                                                                                self.max_input_seq_length)
            input_feat_vec_lengths.append(feat_vec_length)
            # compute sparse tensor inputs
            if len(labels) > self.max_target_seq_length:
                labels = labels[:self.max_target_seq_length]
            indices = [[batch_counter, i] for i in range(len(labels))]
            target_indices += indices
            target_labels += labels
            target_lengths.append(len(labels))
            batch_counter += 1
            # assert len(target_indices) <= self.max_target_seq_length, "target_indices is not less than {0}".format(len(target_indices))
            # assert len(target_labels) <= self.max_target_seq_length, "target_labels is not less than {0}".format(len(target_labels))
            assert len(labels) <= self.max_target_seq_length
            assert len(feat_vec) <= self.max_input_seq_length

        remaining = len(dataset) - (already_processed + num_data_points)
        if remaining == 0:
            batch_pointer = 0
        else:
            batch_pointer += 1
        input_feat_vecs = np.swapaxes(input_feat_vecs, 0, 1)
        if is_train and self.train_conn != None:
            self.train_conn.send([input_feat_vecs, input_feat_vec_lengths,
                                  target_lengths, target_labels, target_indices, batch_pointer])
        elif not is_train and self.test_conn != None:
            self.test_conn.send([input_feat_vecs, input_feat_vec_lengths,
                                 target_lengths, target_labels, target_indices, batch_pointer])
        else:
            return [input_feat_vecs, input_feat_vec_lengths,
                    target_lengths, target_labels, target_indices, batch_pointer]

    def initializeAudioProcessor(self, max_input_seq_length):
        self.audio_processor = audioprocessor.AudioProcessor(max_input_seq_length)

    def setConnections(self, test_conn, train_conn):
        self.train_conn = train_conn
        self.test_conn = test_conn

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
