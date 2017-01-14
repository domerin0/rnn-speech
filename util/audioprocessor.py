# coding=utf-8
import numpy as np
import subprocess
import librosa
import logging


class AudioProcessor(object):
    def __init__(self, max_input_seq_length, feature_type="mfcc"):
        '''
        feature_type - string options are: mfcc, fbank
        mfcc is a 20-dim input 
        fbank is 120-dim input (mel filterbank with delta and double delta)
        '''
        self.max_input_seq_length = max_input_seq_length
        if feature_type == "mfcc":
            self.extraction_f = self.extract_mfcc
            self.feature_size = 20
        elif feature_type == "fbank":
            self.extraction_f = self.extract_fbank
            self.feature_size = 120
        else:
            raise ValueError("{0} is not a valid extraction function, \
            only fbank and mfcc are accepted.".format(feature_type))

    def process_audio_file(self, file_name):
        """
        Reads in audio file, processes it
        Returns padded feature tensor and original length
        """
        sig, sr = librosa.load(file_name, mono=True)
        return self.extraction_f(sig, sr)

    def extract_mfcc(self, sig, sr):
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

    def extract_fbank(self, sig, sr):
        '''
        Compute log mel filterbank features with deltas and double deltas

        This is based on:
        http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html

        TODO energy is not yet obtained.
        '''

        emphasized_signal = np.append(sig[0], sig[1:] - 0.97 * sig[:-1])
        #frame size = 0.025, frame stride = 0.01
        frame_length, frame_step = 0.025 * sr, 0.01 * sr
        signal_length = len(emphasized_signal)
        frame_length = int(round(frame_length))
        frame_step = int(round(frame_step))
        num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

        pad_signal_length = num_frames * frame_step + frame_length
        z = np.zeros((pad_signal_length - signal_length))
        pad_signal = np.append(emphasized_signal, z)

        indices = np.tile(np.arange(0, frame_length),
                             (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step),
                             (frame_length, 1)).T
        frames = pad_signal[indices.astype(np.int32, copy=False)]
        #apply the hamming window function
        frames *= np.hamming(frame_length)
        nfft, nfilt = 512, 40
        mag_frames = np.absolute(np.fft.rfft(frames, nfft))
        pow_frames = ((1.0 / nfft) * ((mag_frames) ** 2))
        low_freq_mel = 0
        high_freq_mel = (2595 * np.log10(1 + (sr / 2) / 700))
        mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
        hz_points = (700 * (10**(mel_points / 2595) - 1))
        bin = np.floor((nfft + 1) * hz_points / sr)

        fbank = np.zeros((nfilt, int(np.floor(nfft / 2 + 1))))
        for m in range(1, nfilt + 1):
            f_m_minus = int(bin[m - 1])   # left
            f_m = int(bin[m])             # center
            f_m_plus = int(bin[m + 1])    # right

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        filter_banks = np.dot(pow_frames, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
        filter_banks = 20 * np.log10(filter_banks)
        #apply mean normalization
        filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)
        filter_banks = filter_banks.transpose()
        delta = librosa.feature.delta(filter_banks)
        double_delta = librosa.feature.delta(delta)
        fbank_feat = np.vstack([filter_banks, delta, double_delta]);

        fbank_feat = fbank_feat.transpose()

        assert np.shape(fbank_feat)[1] == 120, "input dimensions incorrect"

        fbank_length = len(fbank_feat)
        if fbank_length > self.max_input_seq_length:
            fbank_feat = fbank_feat[:self.max_input_seq_length]
        elif fbank_length < self.max_input_seq_length:
            pad_length = self.max_input_seq_length - fbank_length
            #For now length is 120 (feature vector excluding energies)
            padding = np.zeros((pad_length, 120), dtype=np.float)
            fbank_feat = np.concatenate((fbank_feat, padding), 0)
        assert len(fbank_feat) == self.max_input_seq_length, "Padding incorrect..."
        return fbank_feat, fbank_length

    @staticmethod
    def extractWavFromSph(sph_file, wav_file, start, end):
        try:
            subprocess.call(["sox", sph_file, wav_file, "trim", start, "={0}".format(end)])
        except OSError as e:
            logging.warning("Execution failed : %s", e)
            return False
        return True
