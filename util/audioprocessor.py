# coding=utf-8
import numpy as np
import librosa

# GLOBALS
FRAME_STRIDE = 0.01
FRAME_SIZE = 0.025


class AudioProcessor(object):
    def __init__(self, max_input_seq_length, feature_type="mfcc"):
        """
        feature_type - string options are: mfcc, fbank
        mfcc is a 20-dim input 
        fbank is 120-dim input (mel filterbank with delta and double delta)
        """
        self.max_input_seq_length = max_input_seq_length
        self.feature_type = feature_type
        if self.feature_type == "mfcc":
            self._extract_function = self._extract_mfcc
            self.feature_size = 20
        elif self.feature_type == "fbank":
            self._extract_function = self._extract_fbank
            self.feature_size = 120
        else:
            raise ValueError("{0} is not a valid extraction function, \
            only fbank and mfcc are accepted.".format(self.feature_type))

    @staticmethod
    def get_mfcc_length_from_duration(duration):
        """
        Evaluate the mfcc_length for a given file
        Note : returned value is an estimation, librosa will pad so the real size can be bigger (+1 to +3)
        
        :param float duration: duration of the audio file in seconds
        :return int: estimated mfcc_length
        """
        length = int(duration // FRAME_STRIDE) - 1
        return length

    def process_audio_file(self, file_name):
        """
        Reads in audio file, processes it

        :param file_name: an audio file path
        :returns: mfcc: padded feature tensor
        :returns: mfcc_length: original length of the mfcc before padding
        """
        sig, sr = librosa.load(file_name, mono=True)
        return self._extract_function(sig, sr)

    def process_signal(self, sig, sr):
        """
        Reads in audio file, processes it

        :param sig: audio signal to process
        :param sr: audio signal rate
        :returns: mfcc: padded feature tensor
        :returns: mfcc_length: original length of the mfcc before padding
        """
        return self._extract_function(sig, sr)

    def _extract_mfcc(self, sig, sr):
        # mfcc
        mfcc = librosa.feature.mfcc(sig, sr, hop_length=int(round(sr * FRAME_STRIDE)),
                                    n_fft=int(round(sr * FRAME_SIZE)))
        # mfcc is of shape (20 mfcc, time_serie)
        transposed_mfcc = mfcc.transpose()
        mfcc_length = len(transposed_mfcc)

        # Truncate if audio sequence is too long
        if mfcc_length > self.max_input_seq_length:
            transposed_mfcc = transposed_mfcc[:self.max_input_seq_length]

        return transposed_mfcc, mfcc_length

    def _extract_fbank(self, sig, sr):
        """
        Compute log mel filterbank features with deltas and double deltas

        This is based on:
        http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html

        TODO energy is not yet obtained.
        """

        emphasized_signal = np.append(sig[0], sig[1:] - 0.97 * sig[:-1])
        frame_length, frame_step = FRAME_SIZE * sr, FRAME_STRIDE * sr
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
        # Apply the hamming window function
        frames *= np.hamming(frame_length)
        nfft, nfilt = 512, 40
        mag_frames = np.absolute(np.fft.rfft(frames, nfft))
        pow_frames = ((1.0 / nfft) * (mag_frames ** 2))
        low_freq_mel = 0
        
        ### AI:
        # the following line works correcty as-is in python3
        # *** high_freq_mel = (2595 * np.log10(1 + (sr / 2) / 700)) ***
        # however, in python it results in a smaller value of 'high_freq_mel'
        # as the 'sr' variable is interpreted as integer
        # it has minor impact on the performance of the natively trained and tested models
        # however, if one uses python to test the models that were created in python3
        # the models show the CER drop as much as 1% absolute
        # the line below fixes that issue completely
        high_freq_mel = (2595 * np.log10(1 + (float(sr) / 2) / 700))
        
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
        
        # AI:
        # *** filter_banks = 20 * np.log10(filter_banks) ***
        # 'pow_frames' contains the power spectrum (i.e. squared magnitude)
        # the proper formula to convert to the logarithm scale in decibels is
        # 10*log10(POW), while 20*log10(MAG) is used for the un-squared magnitude
        # this way both formuli result in the same outcome
        filter_banks = 10 * np.log10(filter_banks)
        
        # Apply mean normalization
        filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)
        filter_banks = filter_banks.transpose()
        delta = librosa.feature.delta(filter_banks)
        double_delta = librosa.feature.delta(delta)
        fbank_feat = np.vstack([filter_banks, delta, double_delta]);

        fbank_feat = fbank_feat.transpose()

        assert np.shape(fbank_feat)[1] == 120, "input dimensions incorrect"

        # Truncate if audio sequence is too long
        fbank_length = len(fbank_feat)
        if fbank_length > self.max_input_seq_length:
            fbank_feat = fbank_feat[:self.max_input_seq_length]

        return fbank_feat, fbank_length
