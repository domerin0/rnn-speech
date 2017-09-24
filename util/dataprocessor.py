# coding=utf-8
"""
Reads in audio and text data, transforming it for input into the neural nets
It will extract the dataset.
"""
import os
import pickle
import subprocess
import logging
import configparser
from multiprocessing import Pool
import mutagen
import time


DEFAULT_MIN_TEXT_LENGTH = 3         # Default minimum number of chars in a label to be kept into a dataset
DEFAULT_MIN_AUDIO_LENGTH = 0.4      # Default minimum duration (in seconds) of an audio file to be kept into a dataset


class DataProcessor(object):
    def __init__(self, raw_data_paths, file_cache=None, min_text_size=DEFAULT_MIN_TEXT_LENGTH,
                 min_audio_size=DEFAULT_MIN_AUDIO_LENGTH):
        self.raw_data_paths = raw_data_paths.replace(" ", "").split(',')
        self.file_cache = file_cache
        self.min_text_size = min_text_size
        self.min_audio_size = min_audio_size

        # Load the file list
        cached_data = self.load_filelist()
        if cached_data is not None:
            logging.info("{0} : Using audio files list from cache file.".format(self.raw_data_paths))
            self.data = cached_data
        else:
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

            # Adding length
            logging.info("Retrieving audio duration from {0} files. Please wait.".format(len(self.data)))
            start_time = time.time()
            self.data = self._add_audio_length_on_dataset(self.data)
            logging.info("--- Duration : {0}".format(time.time() - start_time))

            # Save the file list if a cache file is provided
            if self.file_cache is not None:
                logging.info("{0} : Saving audio files list to cache file.".format(self.raw_data_paths))
                self.save_filelist(self.data)

        # Check that there is data
        if len(self.data) == 0:
            raise Exception("ERROR : no data found in directories {0}".format(self.raw_data_paths))

        # Filtering small text items
        self.data = [item for item in self.data if len(item[1]) > self.min_text_size]
        # Filtering small files
        self.data = [item for item in self.data if item[2] > self.min_audio_size]

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
    def get_str_labels(char_map, _str):
        """
        Convert a string into a label vector for the model
        The char map follow recommendations from : https://arxiv.org/pdf/1609.05935v2.pdf

        Parameters
        ----------
        :param char_map : the char_map against which to transcode the string
        :param _str : the string to convert into a label

        Returns
        -------
        :return a vector of int
        """
        # add eos char
        _str += char_map[-1]
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
                    result.append(char_map.index(_str[i:i+3]))
                    i += 3
                    continue
                except ValueError:
                    pass
            if len(_str) - i >= 2:
                try:
                    result.append(char_map.index(_str[i:i+2]))
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
                # Add the EOS char and return what was processed
                result.append(len(char_map) - 1)
                return result
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
        files = cls.find_files(raw_data_path, ".trn")
        if files:
            return "Vystadial_2013"
        # Check for ".stm" files
        files = cls.find_files(raw_data_path, ".stm")
        if files:
            return "TEDLIUM"
        # Check for "index.tag.txt" files
        files = cls.find_files(raw_data_path, "index.tags.txt")
        if files:
            return "Shtooka"
        # Check for ".trans.txt" files
        files = cls.find_files(raw_data_path, ".trans.txt")
        if files:
            return "LibriSpeech"
        return "Unrecognized"

    @staticmethod
    def find_files(root_search_path, files_extension):
        files_list = []
        for root, _, files in os.walk(root_search_path):
            files_list.extend([os.path.join(root, file) for file in files if file.endswith(files_extension)])
        return files_list

    @staticmethod
    def _add_audio_length_on_file(audio_file, text, _length):
        file = mutagen.File(audio_file)
        try:
            length = file.info.length
        except AttributeError:
            # In case the type was not recognized by mutagen
            logging.warning("Audio file incorrect : %s", audio_file)
            length = 0
        return [audio_file, text, length]

    @staticmethod
    def _add_audio_length_on_dataset(file_list):
        with Pool() as p:
            result = p.starmap(DataProcessor._add_audio_length_on_file, file_list)
        return result

    def save_filelist(self, data):
        with open(self.file_cache, 'wb') as handle:
            pickle.dump([self.raw_data_paths, data], handle)

    def load_filelist(self):
        if (self.file_cache is not None) and (os.path.exists(self.file_cache)):
            with open(self.file_cache, 'rb') as handle:
                [data_path, data] = pickle.load(handle)
            if data_path == self.raw_data_paths:
                return data
        return None

    def get_data_librispeech(self, raw_data_path):
        text_files = self.find_files(raw_data_path, ".txt")
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
                        result.append([audio_file, self.clean_label(line.replace(head, "")), None])
        return result

    def get_data_shtooka(self, raw_data_path):
        text_files = self.find_files(raw_data_path, ".txt")
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
                        result.append([audio_file, self.clean_label(config[section]['SWAC_TEXT']), None])
        return result

    def get_data_vystadial_2013(self, raw_data_path):
        wav_audio_files = self.find_files(raw_data_path, ".wav")
        # Build from index_tags
        result = []
        for file in wav_audio_files:
            if os.path.exists(file + ".trn"):
                with open(file + ".trn", "r") as f:
                    words = f.readline()
                    result.append([file, self.clean_label(words), None])
        return result

    def get_data_tedlium(self, raw_data_path):
        stm_files = self.find_files(raw_data_path, ".stm")
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
                            result.append([wav_file, self.clean_label(line_list[6]), None])
        return result

    @staticmethod
    def extract_wav_from_sph(sph_file, wav_file, start, end):
        try:
            subprocess.call(["sox", sph_file, wav_file, "trim", start, "={0}".format(end)])
        except OSError as e:
            logging.warning("Execution failed : %s", e)
            return False
        return True
