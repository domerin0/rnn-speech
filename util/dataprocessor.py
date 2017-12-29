# coding=utf-8
"""
Reads in audio and text data, transforming it for input into the neural nets
It will extract the dataset.
"""
import os
import re
import subprocess
import logging
import configparser
import numpy as np


DEFAULT_MIN_TEXT_LENGTH = 3         # Default minimum number of chars in a label to be kept into a dataset
DEFAULT_MIN_AUDIO_LENGTH = 0.4      # Default minimum duration (in seconds) of an audio file to be kept into a dataset


class DataProcessor(object):
    def __init__(self, raw_data_paths, min_text_size=DEFAULT_MIN_TEXT_LENGTH, min_audio_size=DEFAULT_MIN_AUDIO_LENGTH):
        self.raw_data_paths = raw_data_paths.replace(" ", "").split(',')
        self.min_text_size = min_text_size
        self.min_audio_size = min_audio_size

        self.data = []
        for path in self.raw_data_paths:
            data_type = self.get_type(path)
            if data_type == "Shtooka":
                self.data += self.get_data_shtooka(path)
            elif data_type == "Vystadial_2013":
                self.data += self.get_data_vystadial_2013(path)
            elif data_type == "TEDLIUM":
                self.data += self.get_data_tedlium(path)
            elif data_type == "LibriSpeech":
                self.data += self.get_data_librispeech(path)
            else:
                raise Exception("ERROR : unknown training_dataset_type")

        # Check that there is data
        if len(self.data) == 0:
            raise Exception("ERROR : no data found in directories {0}".format(self.raw_data_paths))

        # Filtering small text items
        self.data = [item for item in self.data if len(item[1]) > self.min_text_size]

    def get_dataset(self):
        return self.data

    @staticmethod
    def clean_label(_str):
        """
        Remove unauthorized characters in a string, lower it and remove unneeded spaces

        Parameters
        ----------
        _str : the original string

        Returns
        -------
        string
        """
        _str = _str.strip()
        _str = _str.lower()
        _str = _str.replace(".", "")
        _str = _str.replace(",", "")
        _str = _str.replace("?", "")
        _str = _str.replace("!", "")
        _str = _str.replace(":", "")
        _str = _str.replace("-", " ")
        _str = _str.replace("_", " ")
        _str = _str.replace("  ", " ")
        return _str

    @staticmethod
    def get_str_to_one_hot_encoded(char_map, _str, add_eos=True):
        """
        Convert a string into an array of one-hot encoded vectors

        Parameters
        ----------
        :param char_map : the char_map against which to transcode the string
        :param _str : the string to convert into a label
        :param add_eos : if true (default), add the "end of sentence" special character

        Returns
        -------
        :return: an array of one-hot encoded vectors
        """
        integer_values = DataProcessor.get_str_labels(char_map, _str, add_eos=add_eos)
        result = []
        for integer in integer_values:
            one_hot_vector = np.zeros(len(char_map))
            one_hot_vector[integer] = 1
            result.append(one_hot_vector)
        return result

    @staticmethod
    def get_str_labels(char_map, _str, add_eos=True):
        """
        Convert a string into a label vector for the model
        The char map follow recommendations from : https://arxiv.org/pdf/1609.05935v2.pdf

        Parameters
        ----------
        :param char_map : the char_map against which to transcode the string
        :param _str : the string to convert into a label
        :param add_eos : if true (default), add the "end of sentence" special character

        Returns
        -------
        :return: a vector of int
        """
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
                    result.append(char_map.index(_str[i:i+3].lower()))
                    i += 3
                    continue
                except ValueError:
                    pass
            if len(_str) - i >= 2:
                try:
                    result.append(char_map.index(_str[i:i+2].lower()))
                    i += 2
                    continue
                except ValueError:
                    pass
            try:
                result.append(char_map.index(_str[i:i+1]))
                i += 1
                continue
            except ValueError:
                logging.warning("Unable to process label : %s", _str)
                break
        if add_eos:
            result.append(len(char_map) - 1)
        return result

    @staticmethod
    def get_labels_str(char_map, label):
        """
        Convert a vector issued from the model into a readable string

        Parameters
        ----------
        :param char_map : the char_map against which to transcode the vector
        :param label : a vector of int containing the predicted label

        Returns
        -------
        :return string : the resulting string
        """
        # Convert int to values in self.char_map
        char_list = [char_map[index] for index in label if 0 <= index < len(char_map)]
        # Remove eos character if present
        try:
            char_list.remove(char_map[-1])
        except ValueError:
            pass
        # Add spaces in front of capitalized letters (except the first one) and lower every letter
        result = []
        for i in range(len(char_list)):
            if (i != 0) and (char_list[i].isupper()):
                result.append(" ")
            result.append(char_list[i].lower())
        return "".join(result)

    @classmethod
    def get_type(cls, raw_data_path):
        # Check for ".trn" files
        files = cls.find_files(raw_data_path, "^.*\.trn$")
        if files:
            return "Vystadial_2013"
        # Check for ".stm" files
        files = cls.find_files(raw_data_path, "^.*\.stm$")
        if files:
            return "TEDLIUM"
        # Check for "index.tag.txt" files
        files = cls.find_files(raw_data_path, "^index\.tags\.txt$")
        if files:
            return "Shtooka"
        # Check for ".trans.txt" files
        files = cls.find_files(raw_data_path, "^.*\.trans\.txt$")
        if files:
            return "LibriSpeech"
        return "Unrecognized"

    @staticmethod
    def find_files(root_search_path, pattern):
        files_list = []
        reg_expr = re.compile(pattern)
        for root, _, files in os.walk(root_search_path):
            files_list.extend([os.path.join(root, file) for file in files if reg_expr.match(file)])
        return files_list

    def get_data_librispeech(self, raw_data_path):
        text_files = self.find_files(raw_data_path, "^.*\.txt$")
        result = []
        for text_file in text_files:
            directory = os.path.dirname(text_file)
            with open(text_file, "r") as f:
                lines = f.read().split("\n")
                for line in lines:
                    head = line.split(' ')[0]
                    if len(head) < 5:
                        # Not a line with a file desc
                        break
                    audio_file = directory + "/" + head + ".flac"
                    if os.path.exists(audio_file):
                        result.append([audio_file, self.clean_label(line.replace(head, ""))])
        return result

    def get_data_shtooka(self, raw_data_path):
        text_files = self.find_files(raw_data_path, "^.*\.txt$")
        # Build from index_tags
        result = []
        for file in text_files:
            if file.endswith("index.tags.txt"):
                config = configparser.ConfigParser(comment_prefixes=('#', ';', "\\"))
                config.read(file)
                root = file.replace("index.tags.txt", "")
                for section in config.sections():
                    audio_file = root + section
                    if os.path.exists(audio_file):
                        result.append([audio_file, self.clean_label(config[section]['SWAC_TEXT'])])
        return result

    def get_data_vystadial_2013(self, raw_data_path):
        wav_audio_files = self.find_files(raw_data_path, "^.*\.wav$")
        # Build from index_tags
        result = []
        for file in wav_audio_files:
            if os.path.exists(file + ".trn"):
                with open(file + ".trn", "r") as f:
                    words = f.readline()
                    result.append([file, self.clean_label(words)])
        return result

    def get_data_tedlium(self, raw_data_path):
        stm_files = self.find_files(raw_data_path, "^.*\.stm$")
        # Build from index_tags
        result = []
        for file in stm_files:
            with open(file, "r") as f:
                lines = f.read().split("\n")
                for line in lines:
                    if line == "":
                        continue
                    line_list = line.split(' ', maxsplit=6)
                    if (line_list[2] != "inter_segment_gap") and (line_list[6] != "ignore_time_segment_in_scoring"):
                        start = line_list[3]
                        end = line_list[4]
                        directory = os.path.split(file)[0]
                        sph_file = directory + "/../sph/{0}.sph".format(line_list[0])
                        wav_file = directory + "/../sph/{0}_{1}.wav".format(line_list[0], start)
                        extract_result = None
                        if not os.path.exists(wav_file):
                            extract_result = self.extract_wav_from_sph(sph_file, wav_file, start, end)
                        if extract_result is not False:
                            result.append([wav_file, self.clean_label(line_list[6])])
        return result

    @staticmethod
    def extract_wav_from_sph(sph_file, wav_file, start, end):
        try:
            subprocess.call(["sox", sph_file, wav_file, "trim", start, "={0}".format(end)])
        except OSError as e:
            logging.warning("Execution failed : %s", e)
            return False
        return True
