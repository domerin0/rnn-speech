# coding=utf-8
import numpy as np
import subprocess
import librosa


class AudioProcessor(object):
    def __init__(self, max_input_seq_length):
        self.max_input_seq_length = max_input_seq_length

    def process_audio_file(self, file_name):
        """
        Reads in audio file, processes it
        Returns padded feature tensor and original length
        """
        sig, sr = librosa.load(file_name, mono=True)
        # mfcc
        mfcc = librosa.feature.mfcc(sig, sr)
        # mfcc is of shape (20 mfcc, time_serie)
        transposed_mfcc = mfcc.transpose()
        mfcc_length = len(transposed_mfcc)

        if mfcc_length > self.max_input_seq_length:
            # Audio sequence too long, need to cut
            transposed_mfcc = transposed_mfcc[:self.max_input_seq_length]
        elif mfcc_length < self.max_input_seq_length:
            # Audio sequence too short, need padding to align each sequence
            # (for technical reason, padded part won't be trained or tested)
            pad_length = self.max_input_seq_length - mfcc_length
            padding = np.zeros((pad_length, 20), dtype=np.float)
            transposed_mfcc = np.concatenate((transposed_mfcc, padding), 0)
        assert len(transposed_mfcc) == self.max_input_seq_length, "Padding incorrect..."
        return transposed_mfcc, mfcc_length

    @staticmethod
    def extractWavFromSph(sph_file, wav_file, start, end):
        try:
            subprocess.call("sox {0} {1} trim {2} ={3}".format(sph_file, wav_file, start, end))
        except OSError as e:
            print("Execution failed:", e)
            return False
        return True
