# coding=utf-8
"""
Reads in audio and text data, transforming it for input into the neural nets
It will download and extract the dataset.

TODO generalize it for any audio and corresponding text data

Curently it will only work for one specific dataset.
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
            audio_file_text_pairs, will_convert = self.getFileNameTextPairs_Shtooka(self.raw_data_path)
        elif self.data_type == "Vystadial_2013":
            audio_file_text_pairs, will_convert = self.getFileNameTextPairs_Vystadial_2013(self.raw_data_path)
        elif self.data_type == "TEDLIUM":
            audio_file_text_pairs, will_convert = self.getFileNameTextPairs_TEDLIUM(self.raw_data_path)
        elif self.data_type == "LibriSpeech":
            data_dirs = self.checkWhichDataFoldersArePresent()
            # Check which data folders are present
            if len(data_dirs) == 0:
                raise Exception("ERROR : something went wrong, no data detected, check data directory.")
            # Get pairs of (audio_file_name, transcribed_text)
            audio_file_text_pairs, will_convert = self.getFileNameTextPairs_LibriSpeech(data_dirs)
        else:
            raise Exception("ERROR : unknown training_dataset_type")

        # Check that there is data
        if len(audio_file_text_pairs) == 0:
            raise Exception("ERROR : no data found in directory {0}".format(self.raw_data_path))

        # Shuffle pairs
        shuffle(audio_file_text_pairs)

        if will_convert:
            audio_file_text_pairs_final = []
            for audio_file_name in audio_file_text_pairs:
                if audio_file_name[0].endswith(".flac"):
                    if self.audio_processor.convertAndDeleteFLAC(audio_file_name[0]) is True:
                        audio_file_text_pairs_final.append((audio_file_name[0].replace(".flac", ".wav"),
                                                            audio_file_name[1]))
                    else:
                        # Failed to process the file
                        break
                else:
                    audio_file_text_pairs_final.append((audio_file_name[0], audio_file_name[1]))
        else:
            audio_file_text_pairs_final = audio_file_text_pairs

        return audio_file_text_pairs_final

    def getFileNameTextPairs_LibriSpeech(self, data_dirs):
        """
        Returns
          a list of tuples (audio_file_name, transcribed_text)
          a boolean if some files need to be converted to wav
        """
        audio_file_text_pairs = []
        # Assume there will not need any conversion
        will_convert = False
        for d in data_dirs:
            root_search_path = os.path.join(self.raw_data_path, d)
            flac_audio_files, wav_audio_files, text_files = self.findFiles(root_search_path)
            if len(flac_audio_files) > 0:
                # We have at least one file to convert
                will_convert = True
            audio_files = wav_audio_files + flac_audio_files
            for text_file in text_files:
                with open(text_file, "r") as f:
                    lines = f.read().split("\n")
                    for line in lines:
                        head = line.split(' ')[0]
                        if len(head) < 5:
                            # Not a line with a file desc
                            break
                        matches = next((filepath for filepath in audio_files if filepath.find(head) >= 0), None)
                        if matches is not None:
                            audio_file_text_pairs.append((matches, line.replace(head, "").strip().lower()))

        return audio_file_text_pairs, will_convert

    def checkWhichDataFoldersArePresent(self):
        dirs_to_check = ["dev-clean", "train-other-500", "train-other-100",
                         "train-clean-100", "train-clean-360", "test-clean",
                         "test-other", "dev-other"]
        dirs_available = [name for name in os.listdir(self.raw_data_path)]
        dirs_allowed = []
        for d in dirs_available:
            if d in dirs_to_check:
                dirs_allowed.append(d)
        return dirs_allowed

    @staticmethod
    def findFiles(root_search_path, extension=".txt"):
        flac_audio_files, wav_audio_files, text_files = [], [], []
        for root, _, files in os.walk(root_search_path):
            flac_audio_files.extend([os.path.join(root, audio_file)
                                    for audio_file in files if audio_file.endswith(".flac")])
            wav_audio_files.extend([os.path.join(root, audio_file)
                                   for audio_file in files if audio_file.endswith(".wav")])
            text_files.extend([os.path.join(root, text_file)
                              for text_file in files if text_file.endswith(extension)])
        return flac_audio_files, wav_audio_files, text_files

    def getFileNameTextPairs_Shtooka(self, raw_data_path):
        flac_audio_files, _, text_files = self.findFiles(raw_data_path)
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
                    elif os.path.exists(audio_file.replace(".flac", ".wav")):
                        audio_file_text_pairs.append([audio_file.replace(".flac", ".wav"),
                                                      config[section]['SWAC_TEXT'].strip().lower().replace("_", "-")])
        return audio_file_text_pairs, len(flac_audio_files) > 0

    def getFileNameTextPairs_Vystadial_2013(self, raw_data_path):
        _, wav_audio_files, _ = self.findFiles(raw_data_path)
        # Build from index_tags
        audio_file_text_pairs = []
        for file in wav_audio_files:
            if os.path.exists(file + ".trn"):
                with open(file + ".trn", "r") as f:
                    words = f.readline()
                    audio_file_text_pairs.append([file, words.strip().lower().replace("_", "-")])
        return audio_file_text_pairs, False


    def getFileNameTextPairs_TEDLIUM(self, raw_data_path):
        _, _, stm_files = self.findFiles(raw_data_path, extension=".stm")
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
                        if (not os.path.exists(wav_file)) and (not os.path.exists(wav_file).replace(".wav", ".h5")):
                            extract_result = self.audio_processor.extractWavFromSph(sph_file, wav_file, start, end)
                        if extract_result is not False:
                            audio_file_text_pairs.append([wav_file, line_list[6].strip().lower().replace("_", "-")])
        return audio_file_text_pairs, False

    def filterDataset(self, audio_file_text_pairs, max_input_seq_length, max_target_seq_length):
        new_audio_file_text_pairs = []
        for [file, text] in audio_file_text_pairs:
            if len(text) > max_target_seq_length:
                continue
            _, original_feat_vec_length = self.audio_processor.processFLACAudio(file)
            if original_feat_vec_length > max_input_seq_length:
                continue
            new_audio_file_text_pairs.append([file, text])
        return new_audio_file_text_pairs
