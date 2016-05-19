'''
Not yet implemented!

The model for the character level speech recognizer.

Based on the paper:

http://arxiv.org/pdf/1601.06581v2.pdf

This model is:

character level RNN-LM
'''


class LanguageModel(object):
    def __init__(self, num_labels, num_layers, hidden_size, dropout,
        batch_size, learning_rate,grad_clip, forward_only=False):
        '''
        character level language model to help with acoustic model predictions

        uses lstm cells in a deep rnn

        Inputs:
        num_labels - dimension of character input/one hot encoding
        num_layers - number of lstm layers
        hidden_size - size of hidden layers
        dropout - probability of dropping hidden weights
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
        inputs = tf.placeholder(tf.int32, shape=[None])

        #define cells of character level rnn
        cell = rnn_cell.DropoutWrapper(
        rnn_cell.BasicLSTMCell(hidden_size),
        input_keep_prob=self.dropout_keep_prob_lstm_input,
        output_keep_prob=self.dropout_keep_prob_lstm_output)

        if num_lm_layers > 1:
            cell = rnn.rnn([cell] * num_layers)

        if not forward_only:
            pass
