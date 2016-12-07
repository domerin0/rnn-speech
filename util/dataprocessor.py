# coding=utf-8
"""
Reads in audio and text data, transforming it for input into the neural nets
It will extract the dataset.
"""
import os
from random import shuffle
try:
    import ConfigParser as configparser
except ImportError:
    import configparser


class DataProcessor(object):
    def __init__(self, raw_data_path, data_type, audio_processor):
        self.raw_data_path = raw_data_path
        self.data_type = data_type
        self.audio_processor = audio_processor

    def run(self):
        if self.data_type == "Shtooka":
            audio_file_text_pairs = self.getFileNameTextPairs_Shtooka(self.raw_data_path)
        elif self.data_type == "Vystadial_2013":
            audio_file_text_pairs = self.getFileNameTextPairs_Vystadial_2013(self.raw_data_path)
        elif self.data_type == "TEDLIUM":
            audio_file_text_pairs = self.getFileNameTextPairs_TEDLIUM(self.raw_data_path)
        elif self.data_type == "LibriSpeech":
            audio_file_text_pairs = self.getFileNameTextPairs_LibriSpeech(self.raw_data_path)
        else:
            raise Exception("ERROR : unknown training_dataset_type")

        # Check that there is data
        if len(audio_file_text_pairs) == 0:
            raise Exception("ERROR : no data found in directory {0}".format(self.raw_data_path))

        # Shuffle pairs
        shuffle(audio_file_text_pairs)

        return audio_file_text_pairs

    @staticmethod
    def findFiles(root_search_path, files_extension):
        files_list = []
        for root, _, files in os.walk(root_search_path):
            files_list.extend([os.path.join(root, file) for file in files if file.endswith(files_extension)])
        return files_list

    def getFileNameTextPairs_LibriSpeech(self, raw_data_path):
        text_files = self.findFiles(raw_data_path, ".txt")
        audio_file_text_pairs = []
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
                        audio_file_text_pairs.append([audio_file, line.replace(head, "").strip().lower()])
        return audio_file_text_pairs

    def getFileNameTextPairs_Shtooka(self, raw_data_path):
        text_files = self.findFiles(raw_data_path, ".txt")
        # Build from index_tags
        audio_file_text_pairs = []
        for file in text_files:
            if file.endswith("index.tags.txt"):
                config = configparser.ConfigParser(comment_prefixes=('#', ';', "\\"))
                config.read(file)
                root = file.replace("index.tags.txt", "")
                for section in config.sections():
                    audio_file = root + section
                    if os.path.exists(audio_file):
                        audio_file_text_pairs.append([audio_file,
                                                      config[section]['SWAC_TEXT'].strip().lower().replace("_", "-")])
        return audio_file_text_pairs

    def getFileNameTextPairs_Vystadial_2013(self, raw_data_path):
        wav_audio_files = self.findFiles(raw_data_path, ".wav")
        # Build from index_tags
        audio_file_text_pairs = []
        for file in wav_audio_files:
            if os.path.exists(file + ".trn"):
                with open(file + ".trn", "r") as f:
                    words = f.readline()
                    audio_file_text_pairs.append([file, words.strip().lower().replace("_", "-")])
        return audio_file_text_pairs

    def getFileNameTextPairs_TEDLIUM(self, raw_data_path):
        stm_files = self.findFiles(raw_data_path, ".stm")
        # Build from index_tags
        audio_file_text_pairs = []
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
                            extract_result = self.audio_processor.extractWavFromSph(sph_file, wav_file, start, end)
                        if extract_result is not False:
                            audio_file_text_pairs.append([wav_file, line_list[6].strip().lower().replace("_", "-")])
        return audio_file_text_pairs
