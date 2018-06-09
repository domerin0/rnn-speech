# coding=utf-8
import unittest
from models.SpeechRecognizer import SpeechRecognizer
import tensorflow as tf


class TestSpeechRecognizer(unittest.TestCase):
    model = None
    sess = None

    @classmethod
    def setUpClass(cls):
        with tf.Session():
            cls.model = SpeechRecognizer(language='english')


if __name__ == '__main__':
    unittest.main()
