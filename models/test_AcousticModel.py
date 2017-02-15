# coding=utf-8
import unittest
from models.AcousticModel import AcousticModel
import util.dataprocessor as dataprocessor
import tensorflow as tf


class TestAcousticModel(unittest.TestCase):
    model = None
    sess = None

    @classmethod
    def setUpClass(cls):
        with tf.Session() as sess:
            cls.model = AcousticModel(sess, 2, 50, 0.8, 0.5, 3, 0.0003, 0.33, 5, 1800, 600, 120, False,
                                      forward_only=False, tensorboard_dir=None, tb_run_name=None,
                                      timeline_enabled=False, language='english')

    def test_get_str_labels_and_reverse(self):
        text = "What ! I'm not looking for... I'll do it..."
        cleaned_str = dataprocessor.DataProcessor.clean_label(text)
        numeric_label = self.model.get_str_labels(cleaned_str)
        new_text = self.model.get_labels_str(numeric_label)
        self.assertEqual(new_text, cleaned_str)

    def test_3_chars_token_in_str_end(self):
        text = "it'll"
        cleaned_str = dataprocessor.DataProcessor.clean_label(text)
        numeric_label = self.model.get_str_labels(cleaned_str)
        self.assertEqual(numeric_label, [60, 45, 1, 79])


if __name__ == '__main__':
    unittest.main()
