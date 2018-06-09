# coding=utf-8
import unittest
from models.AcousticModel import AcousticModel
import tensorflow as tf
from models.SpeechRecognizer import ENGLISH_CHAR_MAP


class TestAcousticModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.num_layers = 2
        cls.hidden_size = 50
        cls.batch_size = 1
        cls.max_input_seq_length = 1800
        cls.max_target_seq_length = 600
        cls.input_dim = 120
        cls.normalization = False
        cls.num_labels = 50
        cls.signal_processing = "fbank"
        cls.output_keep_prob = 0.5
        cls.grad_clip = 1
        cls.learning_rate = 0.0003
        cls.lr_decay_factor = 0.33

    def test_create_forward_rnn(self):
        tf.reset_default_graph()
        with tf.Session():
            model = AcousticModel(self.num_layers, self.hidden_size, self.batch_size, self.max_input_seq_length,
                                  self.max_target_seq_length, self.input_dim, self.normalization, self.num_labels)
            model.create_forward_rnn()

    def test_create_training_rnn(self):
        tf.reset_default_graph()
        with tf.Session():
            model = AcousticModel(self.num_layers, self.hidden_size, self.batch_size, self.max_input_seq_length,
                                  self.max_target_seq_length, self.input_dim, self.normalization, self.num_labels)
            model.create_training_rnn(self.output_keep_prob, self.grad_clip,
                                      self.learning_rate, self.lr_decay_factor)

    def test_create_training_rnn_with_iterators(self):
        tf.reset_default_graph()

        with tf.Session():
            model = AcousticModel(self.num_layers, self.hidden_size, self.batch_size, self.max_input_seq_length,
                                  self.max_target_seq_length, self.input_dim, self.normalization, self.num_labels)

            # Create a Dataset from the train_set and the test_set
            train_dataset = model.build_dataset([["/file/path", "Test"]], self.batch_size,
                                                self.max_input_seq_length, self.max_target_seq_length,
                                                self.signal_processing, ENGLISH_CHAR_MAP)
            model.add_dataset_input(train_dataset)
            model.create_training_rnn(self.output_keep_prob, self.grad_clip,
                                      self.learning_rate, self.lr_decay_factor, use_iterator=True)


if __name__ == '__main__':
    unittest.main()
