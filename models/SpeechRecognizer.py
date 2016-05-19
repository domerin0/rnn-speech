'''
Not yet implemented!

The model for the character level speech recognizer.

Based on the paper:

http://arxiv.org/pdf/1601.06581v2.pdf

This model is:

acoustic RNN -> character level RNN-LM
'''

import tensorflow as tf
from tensorflow.models.rnn import seq2seq, rnn_cell, rnn
import tensorflow.contrib.ctc as ctc

class SpeechRecognizer(object):
    def __init__(self):
        '''
        Wrapper class for speech recognizer (combines language
        model and acoustic model)
        '''
        pass
