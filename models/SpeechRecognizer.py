# coding=utf-8
"""
Not yet implemented!

The model for the character level speech recognizer.

Based on the paper:

http://arxiv.org/pdf/1601.06581v2.pdf

This model is:

acoustic RNN -> character level RNN-LM
"""
import util.dataprocessor as dataprocessor
from random import shuffle
import logging
from math import floor


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


class SpeechRecognizer(object):
    def __init__(self, language='english'):
        """
        Wrapper class for speech recognizer (combines language
        model and acoustic model)
        """
        # Set language
        if language == 'english':
            self.char_map = ENGLISH_CHAR_MAP
            self.num_labels = len(self.char_map)
        else:
            raise ValueError("Invalid parameter 'language' for method '__init__'")

    def get_char_map(self):
        return self.char_map

    def get_char_map_length(self):
        return len(self.char_map)

    @staticmethod
    def load_acoustic_dataset(training_dataset_dirs, test_dataset_dirs=None, training_filelist_cache=None,
                              ordered=False, train_frac=None):
        """
        Load the datatsets for the acoustic model training
        Return a train set and an optional test set, each containing a list of [audio_file, label, audio_length]

        Parameters
        ----------
        :param training_dataset_dirs: directory where to find the training data
        :param test_dataset_dirs: directory where to find the test data (optional)
        :param training_filelist_cache: path to the cache file for the training data (optional)
        :param ordered: boolean indicating whether or not to order the dataset by audio files length (ascending)
        :param train_frac: the fraction of the training data to be used as test data
                           (only used if test_dataset_dirs is None)
        :return train_set, test_set: two lists of [audio_file, label, audio_length] where
                                         audio_file is the path to an audio file
                                         label is the true label for the audio file (relative to the char_map)
                                         audio_length if the length of the audio file
        """
        data_processor = dataprocessor.DataProcessor(training_dataset_dirs, file_cache=training_filelist_cache)
        train_set = data_processor.get_dataset()
        if ordered:
            train_set = sorted(train_set, key=lambda x: x[2])
        else:
            shuffle(train_set)
        if test_dataset_dirs is not None:
            # Load the test set data
            data_processor = dataprocessor.DataProcessor(test_dataset_dirs)
            test_set = data_processor.get_dataset()
        elif train_frac is not None:
            # Or use a fraction of the train set for the test set
            num_train = max(1, int(floor(train_frac * len(train_set))))
            test_set = train_set[num_train:]
            train_set = train_set[:num_train]
        else:
            # Or use no test set
            test_set = []

        logging.info("Using %d files in train set", len(train_set))
        logging.info("Using %d size of test set", len(test_set))
        return train_set, test_set
