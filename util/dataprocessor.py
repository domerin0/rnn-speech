'''
Reads in audio and text data, transforming it for input into the neural nets
It will download and extract the dataset.

TODO generalize it for any audio and corresponding text data

Curently it will only work for one specific dataset.
'''
import os
from random import shuffle
import util.audioprocessor as audioprocessor


class DataProcessor(object):
    def __init__(self, raw_data_path):
        self.raw_data_path = raw_data_path
        self.data_dirs = self.checkWhichDataFoldersArePresent()

    def run(self):
        # Check which data folders are present
        if len(self.data_dirs) == 0:
            print("Something went wrong, no data detected, check data directory..")
            return

        # Get pairs of (audio_file_name, transcribed_text)
        print("Figuring out which files need to be processed...")
        audio_file_text_pairs, will_convert = self.getFileNameTextPairs()
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

    def getFileNameTextPairs(self):
        '''
        Returns
          a list of tuples (audio_file_name, transcribed_text)
          a boolean if some files need to be converted to wav
        '''
        audio_file_text_pairs = []
        # Assume there will not need any conversion
        will_convert = False
        for d in self.data_dirs:
            root_search_path = os.path.join(self.raw_data_path, d)
            for root, subdirs, files in os.walk(root_search_path):
                flac_audio_files = [os.path.join(root, audio_file)
                                    for audio_file in files if audio_file.endswith(".flac")]
                wav_audio_files = [os.path.join(root, audio_file)
                                   for audio_file in files if audio_file.endswith(".wav")]
                text_files = [os.path.join(root, text_file) for
                              text_file in files if text_file.endswith(".txt")]
                if len(flac_audio_files) > 0:
                    # We have at least one file to convert
                    will_convert = True
                audio_files = wav_audio_files + flac_audio_files
                if len(audio_files) >= 1 and len(text_files) >= 1:
                    assert len(text_files) == 1, "Issue detected with data directory structure..."
                    with open(text_files[0], "r") as f:
                        lines = f.read().split("\n")
                        for a_file in audio_files:
                            # This might only work on linux
                            audio_file_name = os.path.basename(a_file)
                            head = audio_file_name.replace(".flac", "").replace(".wav", "")
                            for line in lines:
                                if head in line:
                                    text = line.replace(head, "").strip().lower() + "_"
                                    audio_file_text_pairs.append((a_file, text))
                                    break
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
