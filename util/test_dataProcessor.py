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

        # Setup TEDLIUM files
        os.makedirs(self.directory + "TEDLIUM/")
        os.makedirs(self.directory + "TEDLIUM/test/")
        os.makedirs(self.directory + "TEDLIUM/test/stm/")
        text_file = self.directory + "TEDLIUM/test/stm/AimeeMullins_2009P.stm"
        with open(text_file, "w") as f:
            f.write("AimeeMullins_2009P 1 inter_segment_gap 0 17.82 <o,,unknown> ignore_time_segment_in_scoring\n")
            f.write("AimeeMullins_2009P 1 AimeeMullins 17.82 28.81 <o,f0,female> i 'd like to share ...\n")
        # Create empty audio file
        os.makedirs(self.directory + "TEDLIUM/test/sph/")
        open(self.directory + "TEDLIUM/test/sph/AimeeMullins_2009P.sph", 'a').close()

    def tearDown(self):
        shutil.rmtree(self.directory)

    def test_get_type(self):
        data_type = dataprocessor.DataProcessor.get_type(self.directory + "Libri")
        self.assertEqual(data_type, "LibriSpeech")
        data_type = dataprocessor.DataProcessor.get_type(self.directory + "Shtooka")
        self.assertEqual(data_type, "Shtooka")
        data_type = dataprocessor.DataProcessor.get_type(self.directory + "Vystadial_2013")
        self.assertEqual(data_type, "Vystadial_2013")
        data_type = dataprocessor.DataProcessor.get_type(self.directory + "TEDLIUM")
        self.assertEqual(data_type, "TEDLIUM")

    def test_get_data_librispeech(self):
        data_processor = dataprocessor.DataProcessor(self.directory + "Libri", self.audio_processor)
        test_set = data_processor.run()
        self.assertCountEqual(test_set,
                              [[self.directory + "Libri/train-clean-100/19/198/19-198-0000.flac",
                                "northanger abbey", 0],
                               [self.directory + "Libri/train-clean-100/19/198/19-198-0001.flac",
                                "this little work...", 0]
                               ])

    def test_get_data_shtooka(self):
        data_processor = dataprocessor.DataProcessor(self.directory + "Shtooka", self.audio_processor)
        test_set = data_processor.run()
        self.assertCountEqual(test_set,
                              [[self.directory + "Shtooka/flac/eng - I_arose.flac", "i arose", 0],
                               [self.directory + "Shtooka/flac/eng - I_ate.flac", "i ate", 0]
                               ])

    def test_get_data_vystadial_2013(self):
        data_processor = dataprocessor.DataProcessor(self.directory + "Vystadial_2013", self.audio_processor)
        test_set = data_processor.run()
        self.assertCountEqual(test_set,
                              [[self.directory +
                                "Vystadial_2013/data_voip_en/dev/jurcic-028-121024_234433_0013625_0013836.wav",
                                "alright thank you and goodbye", 0]
                               ])

if __name__ == '__main__':
    unittest.main()
