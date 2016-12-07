# coding=utf-8
import unittest
import os
import shutil
import util.audioprocessor as audioprocessor
import util.dataprocessor as dataprocessor


class TestDataProcessor(unittest.TestCase):
    directory = ""
    audio_processor = None

    def setUp(self):
        self.audio_processor = audioprocessor.AudioProcessor(1000)
        # Create a temp dir for testing purpose
        cwd = os.getcwd()
        self.directory = cwd + "/test_directory/"
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        else:
            # Test self.directory already exist, throw an error
            raise Exception('test_directory already exists')
        # Setup LibriSpeech files
        os.makedirs(self.directory + "Libri/")
        os.makedirs(self.directory + "Libri/train-clean-100/")
        os.makedirs(self.directory + "Libri/train-clean-100/" + "19/")
        os.makedirs(self.directory + "Libri/train-clean-100/" + "19/" + "198/")
        text_file = self.directory + "Libri/train-clean-100/19/198/19-198.trans.txt"
        with open(text_file, "w") as f:
            f.write("19-198-0000 NORTHANGER ABBEY\n")
            f.write("19-198-0001 THIS LITTLE WORK...\n")
            f.write("19-198-0002 NEITHER THE...\n")
        # Create empty audio files
        open(self.directory + "Libri/train-clean-100/19/198/19-198-0000.flac", 'a').close()
        open(self.directory + "Libri/train-clean-100/19/198/19-198-0001.flac", 'a').close()

        # Setup Shtooka files
        os.makedirs(self.directory + "Shtooka/")
        os.makedirs(self.directory + "Shtooka/flac/")
        text_file = self.directory + "Shtooka/flac/index.tags.txt"
        with open(text_file, "w") as f:
            f.write("\Swac_Index_Tags\n\n")
            f.write("[GLOBAL]\n")
            f.write("SWAC_LANG = eng\n")
            f.write("SWAC_SPEAK_LANG = eng\n\n")
            f.write("[eng - I_arose.flac]\n")
            f.write("SWAC_TEXT = I arose\n")
            f.write("SWAC_ALPHAIDX = arise\n")
            f.write("SWAC_BASEFORM = arise\n")
            f.write("SWAC_FORM_NAME = Simple Past\n\n")
            f.write("[eng - I_ate.flac]\n")
            f.write("SWAC_TEXT = I ate\n")
            f.write("SWAC_ALPHAIDX = eat\n")
            f.write("SWAC_BASEFORM = eat\n")
            f.write("SWAC_FORM_NAME = Simple Past\n\n")
            f.write("[eng - I_awoke.flac]\n")
            f.write("SWAC_TEXT=I awoke\n")
            f.write("SWAC_ALPHAIDX=awake\n")
            f.write("SWAC_BASEFORM=awake\n")
            f.write("SWAC_FORM_NAME=Simple Past\n")
        # Create empty audio files
        open(self.directory + "Shtooka/flac/eng - I_arose.flac", 'a').close()
        open(self.directory + "Shtooka/flac/eng - I_ate.flac", 'a').close()

        # Setup Vystadial files
        os.makedirs(self.directory + "Vystadial_2013/")
        os.makedirs(self.directory + "Vystadial_2013/data_voip_en/")
        os.makedirs(self.directory + "Vystadial_2013/data_voip_en/dev/")
        text_file = self.directory + "Vystadial_2013/data_voip_en/dev/jurcic-028-121024_234433_0013625_0013836.wav.trn"
        with open(text_file, "w") as f:
            f.write("ALRIGHT THANK YOU AND GOODBYE\n")
        text_file = self.directory + "Vystadial_2013/data_voip_en/dev/jurcic-028-121015_000550_0026689_0027040.wav.trn"
        with open(text_file, "w") as f:
            f.write("FILE WITH NO AUDIO...\n")
        # Create empty audio file
        open(self.directory + "Vystadial_2013/data_voip_en/dev/jurcic-028-121024_234433_0013625_0013836.wav",
             'a').close()

    def tearDown(self):
        shutil.rmtree(self.directory)

    def test_getFileNameTextPairs_LibriSpeech(self):
        data_processor = dataprocessor.DataProcessor(self.directory + "Libri", "LibriSpeech", self.audio_processor)
        test_set = data_processor.run()
        self.assertCountEqual(test_set,
                              [[self.directory + "Libri/train-clean-100/19/198/19-198-0000.flac",
                                "northanger abbey"],
                               [self.directory + "Libri/train-clean-100/19/198/19-198-0001.flac",
                                "this little work..."]
                               ])

    def test_getFileNameTextPairs_Shtooka(self):
        data_processor = dataprocessor.DataProcessor(self.directory + "Shtooka", "Shtooka", self.audio_processor)
        test_set = data_processor.run()
        self.assertCountEqual(test_set,
                              [[self.directory + "Shtooka/flac/eng - I_arose.flac",
                                "i arose"],
                               [self.directory + "Shtooka/flac/eng - I_ate.flac",
                                "i ate"]
                               ])

    def test_getFileNameTextPairs_Vystadial_2013(self):
        data_processor = dataprocessor.DataProcessor(self.directory + "Vystadial_2013",
                                                     "Vystadial_2013", self.audio_processor)
        test_set = data_processor.run()
        self.assertCountEqual(test_set,
                              [[self.directory +
                                "Vystadial_2013/data_voip_en/dev/jurcic-028-121024_234433_0013625_0013836.wav",
                                "alright thank you and goodbye"]
                               ])

if __name__ == '__main__':
    unittest.main()
