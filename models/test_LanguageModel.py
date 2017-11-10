# coding=utf-8
import unittest
from models.LanguageModel import LanguageModel
import tensorflow as tf
from models.SpeechRecognizer import ENGLISH_CHAR_MAP
import numpy as np


class TestLanguageModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.num_layers = 2
        cls.hidden_size = 50
        cls.batch_size = 2
        cls.max_input_seq_length = 1800
        cls.max_target_seq_length = 600
        cls.input_dim = 120
        cls.input_keep_prob = 0.8
        cls.output_keep_prob = 0.5
        cls.grad_clip = 1
        cls.learning_rate = 0.0003
        cls.lr_decay_factor = 0.33

    def test_create_forward_rnn(self):
        tf.reset_default_graph()
        with tf.Session():
            model = LanguageModel(self.num_layers, self.hidden_size, self.batch_size, self.max_input_seq_length,
                                  self.max_target_seq_length, self.input_dim)
            model.create_forward_rnn()

    def test_create_training_rnn(self):
        tf.reset_default_graph()
        with tf.Session():
            model = LanguageModel(self.num_layers, self.hidden_size, self.batch_size, self.max_input_seq_length,
                                  self.max_target_seq_length, self.input_dim)
            model.create_training_rnn(self.input_keep_prob, self.output_keep_prob, self.grad_clip,
                                      self.learning_rate, self.lr_decay_factor)

    def test_build_dataset(self):
        tf.reset_default_graph()

        with tf.Session() as sess:
            model = LanguageModel(self.num_layers, self.hidden_size, self.batch_size, self.max_input_seq_length,
                                  self.max_target_seq_length, self.input_dim)

            # Create a Dataset from the train_set and the test_set
            dataset = model.build_dataset(["the brown lazy fox", "the red quick fox"], self.batch_size,
                                          self.max_input_seq_length, ENGLISH_CHAR_MAP)
            iterator = tf.contrib.data.Iterator.from_dataset(dataset)
            sess.run(iterator.initializer)
            iterator_get_next_op = iterator.get_next()
            input_dataset, input_length_dataset, label_dataset = sess.run(iterator_get_next_op)
            np.testing.assert_array_equal(input_dataset,
                                          [[71, 33, 30, 53, 43, 40, 48, 39, 63, 26, 51, 50, 57, 40, 49, 79],
                                           [71, 33, 30, 69, 30, 29, 68, 46, 34, 28, 36, 57, 40, 49, 79, 0]])
            np.testing.assert_array_equal(input_length_dataset, [18, 17])
            np.testing.assert_array_equal(label_dataset[0],
                                          [[0, 0], [0, 1], [0, 2],  [0, 3],  [0, 4],  [0, 5],  [0, 6],  [0, 7],
                                           [0, 8], [0, 9], [0, 10], [0, 11], [0, 12], [0, 13], [0, 14], [0, 15],
                                           [1, 0], [1, 1], [1, 2],  [1, 3],  [1, 4],  [1, 5],  [1, 6],  [1, 7],
                                           [1, 8], [1, 9], [1, 10], [1, 11], [1, 12], [1, 13], [1, 14]
                                           ])
            np.testing.assert_array_equal(label_dataset[1],
                                          [33, 30, 53, 43, 40, 48, 39, 63, 26, 51, 50, 57, 40, 49, 79, 79,
                                           33, 30, 69, 30, 29, 68, 46, 34, 28, 36, 57, 40, 49, 79, 79])
            np.testing.assert_array_equal(label_dataset[2], [2, 1800])

    def test_create_training_rnn_with_iterators(self):
        tf.reset_default_graph()

        with tf.Session():
            model = LanguageModel(self.num_layers, self.hidden_size, self.batch_size, self.max_input_seq_length,
                                  self.max_target_seq_length, self.input_dim)

            # Create a Dataset from the train_set and the test_set
            train_dataset = model.build_dataset(["the brown lazy fox", "the red quick fox"], self.batch_size,
                                                self.max_input_seq_length)
            model.add_dataset_input(train_dataset)
            model.create_training_rnn(self.input_keep_prob, self.output_keep_prob, self.grad_clip,
                                      self.learning_rate, self.lr_decay_factor, use_iterator=True)


if __name__ == '__main__':
    unittest.main()
