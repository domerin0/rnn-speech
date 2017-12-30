# coding=utf-8
import unittest
import os
import shutil
import util.dataprocessor as dataprocessor
from models.SpeechRecognizer import ENGLISH_CHAR_MAP
import numpy as np


class TestDataProcessor(unittest.TestCase):
    directory = ""

    @classmethod
    def setUpClass(cls):
        # Create a temp dir for testing purpose
        cwd = os.getcwd()
        cls.directory = cwd + "/test_directory/"
        if not os.path.exists(cls.directory):
            os.makedirs(cls.directory)
        else:
            # Test self.directory already exist, throw an error
            raise Exception('test_directory already exists')
        # Setup LibriSpeech files
        os.makedirs(cls.directory + "Libri/")
        os.makedirs(cls.directory + "Libri/train-clean-100/")
        os.makedirs(cls.directory + "Libri/train-clean-100/" + "19/")
        os.makedirs(cls.directory + "Libri/train-clean-100/" + "19/" + "198/")
        text_file = cls.directory + "Libri/train-clean-100/19/198/19-198.trans.txt"
        with open(text_file, "w") as f:
            f.write("19-198-0000 NORTHANGER ABBEY\n")
            f.write("19-198-0001 THIS LITTLE WORK...\n")
            f.write("19-198-0002 NEITHER THE...\n")
        # Create empty audio files
        open(cls.directory + "Libri/train-clean-100/19/198/19-198-0000.flac", 'a').close()
        open(cls.directory + "Libri/train-clean-100/19/198/19-198-0001.flac", 'a').close()

        # Setup Shtooka files
        os.makedirs(cls.directory + "Shtooka/")
        os.makedirs(cls.directory + "Shtooka/flac/")
        text_file = cls.directory + "Shtooka/flac/index.tags.txt"
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
        open(cls.directory + "Shtooka/flac/eng - I_arose.flac", 'a').close()
        open(cls.directory + "Shtooka/flac/eng - I_ate.flac", 'a').close()

        # Setup Vystadial files
        os.makedirs(cls.directory + "Vystadial_2013/")
        os.makedirs(cls.directory + "Vystadial_2013/data_voip_en/")
        os.makedirs(cls.directory + "Vystadial_2013/data_voip_en/dev/")
        text_file = cls.directory + "Vystadial_2013/data_voip_en/dev/jurcic-028-121024_234433_0013625_0013836.wav.trn"
        with open(text_file, "w") as f:
            f.write("ALRIGHT THANK YOU AND GOODBYE\n")
        text_file = cls.directory + "Vystadial_2013/data_voip_en/dev/jurcic-028-121015_000550_0026689_0027040.wav.trn"
        with open(text_file, "w") as f:
            f.write("FILE WITH NO AUDIO...\n")
        # Create empty audio file
        open(cls.directory + "Vystadial_2013/data_voip_en/dev/jurcic-028-121024_234433_0013625_0013836.wav",
             'a').close()

        # Setup TEDLIUM files
        os.makedirs(cls.directory + "TEDLIUM/")
        os.makedirs(cls.directory + "TEDLIUM/test/")
        os.makedirs(cls.directory + "TEDLIUM/test/stm/")
        text_file = cls.directory + "TEDLIUM/test/stm/AimeeMullins_2009P.stm"
        with open(text_file, "w") as f:
            f.write("AimeeMullins_2009P 1 inter_segment_gap 0 17.82 <o,,unknown> ignore_time_segment_in_scoring\n")
            f.write("AimeeMullins_2009P 1 AimeeMullins 17.82 28.81 <o,f0,female> i 'd like to share ...\n")
        # Create empty audio file
        os.makedirs(cls.directory + "TEDLIUM/test/sph/")
        open(cls.directory + "TEDLIUM/test/sph/AimeeMullins_2009P.sph", 'a').close()

        # Setup VCTK files
        os.makedirs(cls.directory + "VCTK/")
        os.makedirs(cls.directory + "VCTK/wav48/")
        os.makedirs(cls.directory + "VCTK/wav48/p225/")
        os.makedirs(cls.directory + "VCTK/txt/")
        os.makedirs(cls.directory + "VCTK/txt/p225/")
        text_file = cls.directory + "VCTK/txt/p225/p225_001.txt"
        with open(text_file, "w") as f:
            f.write("Please call Stella.\n")
        text_file = cls.directory + "VCTK/txt/p225/p225_002.txt"
        with open(text_file, "w") as f:
            f.write("Ask her to bring these things with her from the store.")
        # Create empty audio files
        open(cls.directory + "VCTK/wav48/p225/p225_001.wav", 'a').close()
        open(cls.directory + "VCTK/wav48/p225/p225_002.wav", 'a').close()


    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.directory)

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
        data_processor = dataprocessor.DataProcessor(self.directory + "Libri")
        test_set = data_processor.get_dataset()
        self.assertCountEqual(test_set,
                              [[self.directory + "Libri/train-clean-100/19/198/19-198-0000.flac",
                                "northanger abbey"],
                               [self.directory + "Libri/train-clean-100/19/198/19-198-0001.flac",
                                "this little work"]
                               ])

    def test_get_data_shtooka(self):
        data_processor = dataprocessor.DataProcessor(self.directory + "Shtooka")
        test_set = data_processor.get_dataset()
        self.assertCountEqual(test_set,
                              [[self.directory + "Shtooka/flac/eng - I_arose.flac", "i arose"],
                               [self.directory + "Shtooka/flac/eng - I_ate.flac", "i ate"]
                               ])

    def test_get_data_vystadial_2013(self):
        data_processor = dataprocessor.DataProcessor(self.directory + "Vystadial_2013")
        test_set = data_processor.get_dataset()
        self.assertCountEqual(test_set,
                              [[self.directory +
                                "Vystadial_2013/data_voip_en/dev/jurcic-028-121024_234433_0013625_0013836.wav",
                                "alright thank you and goodbye"]
                               ])

    def test_get_data_vctk(self):
        data_processor = dataprocessor.DataProcessor(self.directory + "VCTK")
        test_set = data_processor.get_dataset()
        self.assertCountEqual(test_set,
                              [[self.directory + "VCTK/wav48/p225/p225_001.wav",
                                "please call stella"],
                               [self.directory + "VCTK/wav48/p225/p225_002.wav",
                               "ask her to bring these things with her from the store"]]
                              )

    def test_get_str_labels_and_reverse(self):
        text = "What ! I'm not looking for... I'll do it..."
        cleaned_str = dataprocessor.DataProcessor.clean_label(text)
        numeric_label = dataprocessor.DataProcessor.get_str_labels(ENGLISH_CHAR_MAP, cleaned_str)
        new_text = dataprocessor.DataProcessor.get_labels_str(ENGLISH_CHAR_MAP, numeric_label)
        self.assertEqual(new_text, cleaned_str)

    def test_3_chars_token_in_str_end(self):
        text = "it'll"
        cleaned_str = dataprocessor.DataProcessor.clean_label(text)
        numeric_label = dataprocessor.DataProcessor.get_str_labels(ENGLISH_CHAR_MAP, cleaned_str)
        self.assertEqual(numeric_label, [60, 45, 1, 79])

    def test_first_value_in_char_map(self):
        text = "'d"
        cleaned_str = dataprocessor.DataProcessor.clean_label(text)
        numeric_label = dataprocessor.DataProcessor.get_str_labels(ENGLISH_CHAR_MAP, cleaned_str)
        self.assertEqual(numeric_label, [0, 79])

    def test_get_str_to_one_hot_encoded_first_item(self):
        vector = dataprocessor.DataProcessor.get_str_to_one_hot_encoded(ENGLISH_CHAR_MAP, "'d")
        np.testing.assert_array_equal(vector, [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 0.],
                                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 1.]
                                               ])

    def test_get_str_to_one_hot_encoded_last_item(self):
        vector = dataprocessor.DataProcessor.get_str_to_one_hot_encoded(ENGLISH_CHAR_MAP, "_", add_eos=False)
        np.testing.assert_array_equal(vector, [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 1.]
                                               ])

    def test_get_str_to_one_hot_encoded_double_letter(self):
        vector = dataprocessor.DataProcessor.get_str_to_one_hot_encoded(ENGLISH_CHAR_MAP, "bb", add_eos=False)
        np.testing.assert_array_equal(vector, [[0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
                                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 0.]
                                               ])

    def test_get_str_to_one_hot_encoded_full_string(self):
        vector = dataprocessor.DataProcessor.get_str_to_one_hot_encoded(ENGLISH_CHAR_MAP, "i will")
        # This will be encoded to "IWill_" with "ll" being a specific letter
        np.testing.assert_array_equal(vector, [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
                                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 0.],
                                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
                                                0., 0.],
                                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
                                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 0.],
                                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 0.],
                                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                0., 1.]
                                               ])


if __name__ == '__main__':
    unittest.main()
