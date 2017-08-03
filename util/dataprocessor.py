# coding=utf-8
"""
Reads in audio and text data, transforming it for input into the neural nets
It will extract the dataset.
"""
import os
import pickle
import subprocess
import logging
from random import shuffle
from util.audioprocessor import AudioProcessor
try:
    import ConfigParser as configparser
except ImportError:
    import configparser


class DataProcessor(object):
    def __init__(self, raw_data_paths, file_cache=None, size_ordering='False', min_text_size=3, min_audio_size=40):
        self.raw_data_paths = raw_data_paths.replace(" ", "").split(',')
        self.size_ordering = size_ordering
        self.file_cache = file_cache
        self.min_text_size = min_text_size
        self.min_audio_size = min_audio_size

    def run(self):
        # Load the file list
        data = self.load_filelist()
        if data is not None:
            logging.info("{0} : Using audio files list from cache file.".format(self.raw_data_paths))
        else:
            data = []
            for path in self.raw_data_paths:
                data_type = self.get_type(path)
                if data_type == "Shtooka":
                    data += self.get_data_shtooka(path)
                elif data_type == "Vystadial_2013":
                    data += self.get_data_vystadial_2013(path)
                elif data_type == "TEDLIUM":
                    data += self.get_data_tedlium(path)
                elif data_type == "LibriSpeech":
                    data += self.get_data_librispeech(path)
                else:
                    raise Exception("ERROR : unknown training_dataset_type")

            # Adding length
            if self.size_ordering == 'True' or self.size_ordering == 'First_run_only':
                data = self.add_audio_file_length(data)

            # Save the file list if a cache file is provided
            if self.file_cache is not None:
                logging.info("{0} : Saving audio files list to cache file.".format(self.raw_data_paths))
                self.save_filelist(data)

        # Check that there is data
        if len(data) == 0:
            raise Exception("ERROR : no data found in directories {0}".format(self.raw_data_paths))

        # Order by size ascending if needed
        if self.size_ordering == 'True' or self.size_ordering == 'First_run_only':
            # Check that size is present in the list (in case the cache file was produced without it)
            if data[0][2] is None:
                raise Exception("Cache file do not have files' length,"
                                "please remove the cache file : {0}".format(self.file_cache))
            logging.debug("{0} : Sorting the audio files list by duration".format(self.raw_data_paths))
            data = sorted(data, key=lambda data: data[2])
        else:
            logging.debug("{0} : Shuffling the audio files list".format(self.raw_data_paths))
            shuffle(data)

        # Filtering small text items
        data = [item for item in data if len(item[1]) > self.min_text_size]
        # Filtering small files if we have the size
        data = [item for item in data if (item[2] is None) or (item[2] > self.min_audio_size)]

        return data

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
    def add_audio_file_length(file_list):
        logging.info("Getting audio files duration, this could take long (%d files to process)", len(file_list))

        result = []
        previous_percent = 0
        for index, [audio_file, text, _] in enumerate(file_list):
            length = AudioProcessor.get_audio_file_length(audio_file)
            new_percent = int(round((index / len(file_list)) * 100))
            if new_percent != previous_percent:
                logging.info("%d %% done", new_percent)
                previous_percent = new_percent
            result.append([audio_file, text, length])

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
