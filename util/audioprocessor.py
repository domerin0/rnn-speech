# coding=utf-8
"""
a 40-dimensional log mel-frequency
filterbank feature vector with energy and their delta and double-delta
values

The feature vectors
are extracted every 10 ms with 25 ms Hamming window
"""
from python_speech_features import fbank
import scipy.io.wavfile as wav
import numpy as np
import os
import subprocess
import h5py


class AudioProcessor(object):
    def __init__(self, max_input_seq_length, load_save_input_vec=False):
        self.max_input_seq_length = max_input_seq_length
        self.load_save_input_vec = load_save_input_vec

    def processFLACAudio(self, wav_file_name):
        """
        Reads in audio file, processes it
        Returns padded feature tensor and unpadded length
        """
        # Check if computed input vector already exists
        if self.load_save_input_vec and os.path.exists(wav_file_name.replace(".wav", ".h5")):
            with h5py.File(wav_file_name.replace(".wav", ".h5"), 'r') as hf:
                try:
                    feat_vec = hf['feat_vec'].value
                    original_feat_vec_length = hf['original_feat_vec_length'].value
                except:
                    print("Error loading file : ", wav_file_name.replace(".wav", ".h5"))
                    return [], 0
        else:
            feat_vec = self.computeLogMelFilterBank(wav_file_name)
            original_feat_vec_length = len(feat_vec)
            # Save the file if needed
            if self.load_save_input_vec:
                with h5py.File(wav_file_name.replace(".wav", ".h5"), 'w') as hf:
                    hf.create_dataset('feat_vec', data=feat_vec)
                    hf.create_dataset('original_feat_vec_length', data=original_feat_vec_length)

        if original_feat_vec_length > self.max_input_seq_length:
            # Audio sequence too long, need to cut
            feat_vec = feat_vec[:self.max_input_seq_length]
        elif original_feat_vec_length < self.max_input_seq_length:
            # Audio sequence too short, need padding to align each sequence
            # (for technical reason, padded part won't be trained or tested)
            pad_length = self.max_input_seq_length - original_feat_vec_length
            padding = np.zeros((pad_length, 123), dtype=np.float)
            feat_vec = np.concatenate((feat_vec, padding), 0)
        assert len(feat_vec) == self.max_input_seq_length, "Padding incorrect..."
        return feat_vec, original_feat_vec_length

    def convertAndDeleteFLAC(self, audio_file_name):
        self.convertFlac2Wav(audio_file_name)
        self.deleteWav(audio_file_name)

    def computeLogMelFilterBank(self, file_name):
        """
        Compute the log-mel frequency filterbank feature vector with deltas and
        double deltas
        """
        (rate, sig) = wav.read(file_name)
        fbank_feat, energy = fbank(sig, rate, winlen=0.025, winstep=0.01, nfilt=40)
        fbank_feat = np.log(fbank_feat)
        fbank_feat = np.vstack((fbank_feat.transpose(), energy.transpose())).transpose()
        deltas = self.computeDeltas(fbank_feat)
        assert deltas.shape == fbank_feat.shape, "Shapes not equal {0} and \
        {1}".format(deltas.shape, fbank_feat.shape)
        feat_vec = np.vstack((fbank_feat.transpose(), deltas.transpose()))
        double_deltas = self.computeDeltas(deltas)
        feat_vec = np.vstack((feat_vec, double_deltas.transpose())).transpose()
        assert len(feat_vec[0]) == 123, "Something wrong with feature vector dimensions..."
        return feat_vec

    @staticmethod
    def computeDeltas(fbank_frames, N=2):
        """
        Implementation of this based on formula found at:
        http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
        """
        frames = []
        for i, feat_vec in enumerate(fbank_frames):
            deltas = []
            # If there are not enough side frames, we put
            # 0s in for the deltas
            if i - N < 0 or i + N > (len(fbank_frames) - 1):
                frames.append([0] * len(feat_vec))
            else:
                for ii in range(len(feat_vec)):
                    top_sum, bottom_sum = 0.0, 0.0
                    for n in range(1, N+1):
                        top_sum += n * (fbank_frames[i+n][ii] -
                                        fbank_frames[i-n][ii])
                        bottom_sum += n*n
                    deltas.append(top_sum / (2.0 * bottom_sum))
                frames.append(deltas)
        return np.array(frames)

    @staticmethod
    def convertFlac2Wav(file_name):
        """
        Convert the flac file to wav (so we can process on it)
        """
        try:
            subprocess.call("sox {0} {1}".format(file_name, file_name.replace(".flac", ".wav")))
        except OSError as e:
            print("Execution failed:", e)
        return file_name.replace(".flac", ".wav")

    @staticmethod
    def deleteWav(file_name):
        """
        Delete wav file after we're done with it
        """
        if file_name.endswith(".flac"):
            os.remove(file_name)

    @staticmethod
    def extractWavFromSph(sph_file, wav_file, start, end):
        try:
            subprocess.call("sox {0} {1} trim {2} ={3}".format(sph_file, wav_file, start, end))
        except OSError as e:
            print("Execution failed:", e)
            return False
        return True

