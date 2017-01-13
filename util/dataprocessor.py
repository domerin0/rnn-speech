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
    def __init__(self, raw_data_paths, audio_processor, size_ordering=False):
        self.raw_data_paths = raw_data_paths.replace(" ", "").split(',')
        self.audio_processor = audio_processor
        self.size_ordering = size_ordering

    def run(self):
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

        # Check that there is data
        if len(data) == 0:
            raise Exception("ERROR : no data found in directories {0}".format(self.raw_data_paths))

        # Order by size ascending if needed
        if self.size_ordering is True:
            data = sorted(data, key=lambda data: data[2])
        else:
            shuffle(data)

        return data

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
                        file_size = os.path.getsize(audio_file)
                        result.append([audio_file, line.replace(head, "").strip().lower(), file_size])
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
                        file_size = os.path.getsize(audio_file)
                        result.append([audio_file, config[section]['SWAC_TEXT'].strip().lower().replace("_", "-"),
                                      file_size])
        return result

    def get_data_vystadial_2013(self, raw_data_path):
        wav_audio_files = self.find_files(raw_data_path, ".wav")
        # Build from index_tags
        result = []
        for file in wav_audio_files:
            if os.path.exists(file + ".trn"):
                with open(file + ".trn", "r") as f:
                    words = f.readline()
                    file_size = os.path.getsize(file)
                    result.append([file, words.strip().lower().replace("_", "-"), file_size])
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
                            extract_result = self.audio_processor.extractWavFromSph(sph_file, wav_file, start, end)
                        if extract_result is not False:
                            file_size = os.path.getsize(wav_file)
                            result.append([wav_file, line_list[6].strip().lower().replace("_", "-"), file_size])
        return result
