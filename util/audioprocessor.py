'''
a 40-dimensional log mel-frequency
filterbank feature vector with energy and their delta and double-delta
values

The feature vectors
are extracted every 10 ms with 25 ms Hamming window
'''
from features import mfcc
from features import logfbank
from features import fbank
import scipy.io.wavfile as wav
import numpy as np
import os

class AudioProcessor(object):
    def __init__(self, max_input_seq_length):
        self.master_file_list = "master_file_list.txt"
        self.max_input_seq_length = max_input_seq_length

    def run(self, file_name_text_pairs, output_dir, thread_num):
        with open(os.path.join(output_dir, self.master_file_list), "a+") as f:
            counter = 0
            for file_name in file_name_text_pairs[0]:
                counter += 1
                if counter % 100 == 0:
                    print "Processing file {0}... on thread {1}".format(counter,
                        thread_num)
                if len(file_name[1]) > self.max_target_seq_length:
                    continue
                wav_file = self.convertFlac2Wav(file_name[0])
                feat_vec = self.computeLogMelFilterBank(wav_file)
                if len(feat_vec) > self.max_input_seq_length:
                    print "skipping"
                    continue
                elif len(feat_vec) < self.max_input_seq_length:
                    pad_length = self.max_input_seq_length - len(feat_vec)
                    padding = np.zeros((pad_length, 123), dtype=np.float)
                    feat_vec = np.concatenate((feat_vec, padding), 0)
                    assert len(feat_vec) == self.max_input_seq_length, "Padding incorrect..."
                self.deleteWav(wav_file)
                #save feature vector
                array_file_name = os.path.basename(file_name[0]).replace(".flac", ".npy")
                np.save(os.path.join(output_dir, array_file_name), feat_vec)
                f.write("{0}, {1}\n".format(array_file_name,
                    file_name[1]))

    def processFLACAudio(self, wav_file_name):
        '''
        Reads in audio file, processes it
        Returns padded feature tensor and unpadded length
        '''
        feat_vec = self.computeLogMelFilterBank(wav_file_name)
        feat_vec_length = len(feat_vec)
        if feat_vec_length > self.max_input_seq_length:
            feat_vec = feat_vec[:self.max_input_seq_length]
            feat_vec_length = len(feat_vec)
        elif feat_vec_length <= self.max_input_seq_length:
            pad_length = self.max_input_seq_length - len(feat_vec)
            padding = np.zeros((pad_length, 123), dtype=np.float)
            feat_vec = np.concatenate((feat_vec, padding), 0)
        assert len(feat_vec) == self.max_input_seq_length, "Padding incorrect..."
        return feat_vec, feat_vec_length

    def convertAndDeleteFLAC(self, audio_file_name):
        wav_file = self.convertFlac2Wav(audio_file_name)
        self.deleteWav(audio_file_name)

    def computeLogMelFilterBank(self, file_name):
        '''
        Compute the log-mel frequency filterbank feature vector with deltas and
        double deltas
        '''
        (rate, sig) = wav.read(file_name)
        fbank_feat, energy = fbank(sig,rate, winlen=0.025,winstep=0.01, nfilt=40)
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

    def computeDeltas(self, fbank_feat, N=2):
        '''
        Implementation of this based on formula found at:
        http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
        '''
        deltas = []
        for feat_vec in fbank_feat:
            row = []
            for i in range(len(feat_vec)):
                top_sum, bottom_sum = 0.0, 0.0
                for n in range(1, N+1):
                    if i-n < 0:
                        top_sum += n * (feat_vec[i + n])
                    elif i+n >= len(feat_vec):
                        top_sum += n * (feat_vec[i - n])
                    else:
                        top_sum += n * (feat_vec[i + n] - feat_vec[i - n])
                    bottom_sum += n*n
                row.append(top_sum / (2.0 * bottom_sum))
            deltas.append(row)
        return np.array(deltas)

    def convertFlac2Wav(self, file_name):
        '''
        Convert the flac file to wav (so we can process on it)
        '''
        os.system("sox {0} {1}".format(file_name,
            file_name.replace(".flac", ".wav")))
        return file_name.replace(".flac", ".wav")

    def deleteWav(self, file_name):
        '''
        Delete wav file after we're done with it
        '''
        if file_name.endswith(".flac"):
            os.remove(file_name)
