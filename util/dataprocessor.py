# coding=utf-8
"""
Reads in audio and text data, transforming it for input into the neural nets
It will download and extract the dataset.

TODO generalize it for any audio and corresponding text data

Curently it will only work for one specific dataset.
"""
import os
from random import shuffle
import util.audioprocessor as audioprocessor
try:
    import ConfigParser as configparser
except ImportError:
    import configparser

class DataProcessor(object):
    def __init__(self, raw_data_path, data_type):
        self.raw_data_path = raw_data_path
        self.data_type = data_type

    def run(self):
        print("Figuring out which files need to be processed...")
        if self.data_type == "Shtooka":
            audio_file_text_pairs, will_convert = self.getFileNameTextPairs_Shtooka(self.raw_data_path)
        elif self.data_type == "LibriSpeech":
            data_dirs = self.checkWhichDataFoldersArePresent()
            # Check which data folders are present
            if len(data_dirs) == 0:
                print("Something went wrong, no data detected, check data directory..")
                return
            # Get pairs of (audio_file_name, transcribed_text)
            audio_file_text_pairs, will_convert = self.getFileNameTextPairs_LibriSpeech(data_dirs)
        else:
            raise Exception("unknown training_dataset_type")

        print("Using {0} files in total dataset...".format(len(audio_file_text_pairs)))
        # Shuffle pairs
        shuffle(audio_file_text_pairs)

        if will_convert:
            audio_file_text_pairs_final = []
            audio_processor = audioprocessor.AudioProcessor(1)
            for audio_file_name in audio_file_text_pairs:
                if audio_file_name[0].endswith(".flac"):
                    audio_processor.convertAndDeleteFLAC(audio_file_name[0])
                    audio_file_text_pairs_final.append((audio_file_name[0].replace(".flac", ".wav"),
                                                        audio_file_name[1]))
                else:
                    audio_file_text_pairs_final.append((audio_file_name[0], audio_file_name[1]))
        else:
            audio_file_text_pairs_final = audio_file_text_pairs

        return audio_file_text_pairs_final

    def getFileNameTextPairs_LibriSpeech(self, data_dirs):
        '''
        Returns
          a list of tuples (audio_file_name, transcribed_text)
          a boolean if some files need to be converted to wav
        '''
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
    def findFiles(root_search_path):
        flac_audio_files, wav_audio_files, text_files = [], [], []
        for root, _, files in os.walk(root_search_path):
            flac_audio_files.extend([os.path.join(root, audio_file)
                                    for audio_file in files if audio_file.endswith(".flac")])
            wav_audio_files.extend([os.path.join(root, audio_file)
                                   for audio_file in files if audio_file.endswith(".wav")])
            text_files.extend([os.path.join(root, text_file)
                              for text_file in files if text_file.endswith(".txt")])
        return flac_audio_files, wav_audio_files, text_files

    def getFileNameTextPairs_Shtooka(self, raw_data_path):
        config = configparser.ConfigParser(comment_prefixes=('#', ';', "\\"))
        flac_audio_files, wav_audio_files, text_files = self.findFiles(raw_data_path)
        audio_files = wav_audio_files + flac_audio_files
        # Build from index_tags
        audio_file_text_pairs = []
        for file in text_files:
            if file.endswith("index.tags.txt"):
                config = configparser.ConfigParser(comment_prefixes=('#', ';', "\\"))
                config.read(file)
                for section in config.sections():
                    audio_file = [audio_file for audio_file in audio_files if audio_file.endswith(section) or
                                  audio_file.endswith(section.replace(".flac", ".wav"))]
                    if len(audio_file) > 0:
                        audio_file_text_pairs.append([audio_file[0],
                                                      config[section]['SWAC_TEXT'].strip().lower().replace("_", "-")])
        return audio_file_text_pairs, len(flac_audio_files) > 0

