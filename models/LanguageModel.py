'''
Not yet implemented!

The model for the character level speech recognizer.

Based on the paper:

http://arxiv.org/pdf/1601.06581v2.pdf

This model is:

character level RNN-LM
'''


class LanguageModel(object):
    def __init__(self, num_labels, seq_length,num_layers,
                 hidden_size, dropout, batch_size, learning_rate,
                 grad_clip, forward_only=False):
        '''
        character level language model to help with acoustic model predictions

        uses lstm cells in a deep rnn

        Inputs:
        num_labels - dimension of character input/one hot encoding
        seq_length - length of text sequence to train
        num_layers - number of lstm layers
        hidden_size - size of hidden layers
        dropout - probability of keeping hidden weights
        batch_size - number of training examples fed at once
        learning_rate - learning rate parameter fed to optimizer
        '''
        self.dropout = dropout
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
        	self.learning_rate * lr_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)
        self.dropout_keep_prob_lstm_input = tf.constant(self.dropout)
        self.dropout_keep_prob_lstm_output = tf.constant(self.dropout)

        #input labels
        #inputs is a tensor of one-hot encoded vectors
        self.inputs = tf.placeholder(tf.int32,
                                    shape=[seq_length, None, input_dim],
                                    name="inputs")

        #define cells of character level rnn
        cell = rnn_cell.DropoutWrapper(
            rnn_cell.BasicLSTMCell(hidden_size),
            input_keep_prob=self.dropout_keep_prob_lstm_input,
            output_keep_prob=self.dropout_keep_prob_lstm_output)

        if num_lm_layers > 1:
            cell = rnn.rnn([cell] * num_layers)

        #build input layer
        w_i = tf.get_variable("input_w", [input_dim, hidden_size])
        b_i = tf.get_variable("input_b", [hidden_size])

        inputs = [tf.matmul(tf.squeeze(i, squeeze_dims=[0]), w_i) + b_i
                  for i in tf.split(0, seq_length, self.inputs)]

        # set rnn init state to 0s
        initial_state = cell.zero_state(self.batch_size, tf.float32)

        #build rnn
        rnn_output, hidden_state = rnn.rnn(cell, inputs,
                                            initial_state=initial_state)

        # build output layer
        w_o = tf.get_variable("output_w", [hidden_size, num_labels])
        b_o = tf.get_variable("output_b", [num_labels])


        # compute logits
        self.logits = [tf.matmul(tf.squeeze(i, squeeze_dims=[0]), w_o) + b_o
                       for i in tf.split(0, self.max_input_seq_length, rnn_output)]

        if forward_only:
            pass
        else:
            self.targets = tf.placeholder(tf.int32,
                                        shape=[seq_length, None, input_dim],
                                        name="targets")

            self.losses =tf.nn.softmax_cross_entropy_with_logits(self.logits,
                                                                 self.target,
                                                                 name="losses")

    def get_batch():
        pass

    def step():
        pass

    def process_input():
        pass

    def get_num_batches():
        pass

    def train():
        pass
