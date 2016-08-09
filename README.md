# rnn-speech
Character level speech recognizer using ctc loss with deep rnns in TensorFlow.

###About

This is an ongoing project, working towards an implementation of the charater-level ISR detailed in the [paper](http://arxiv.org/pdf/1601.06581v2.pdf)
by Kyuyeon Hwang and Wonyong Sung. It works at the character level using 1 deep rnn trained with ctc loss for the acoustic model, and one deep rnn trained for a character-level language model. The acoustic model reads in log mel frequency filterbank feature vectors with energy, delta and delta-delta values (123-dim inputs).

The audio signal processing is done using jameslyons' [python_speech_features](https://github.com/jameslyons/python_speech_features),
and this [MFCC tutorial](http://www.practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/).

Currently only the acoustic model has been (mostly) completed. The character-level RNN-LM is in the works. There is not as of yet any way to sample it using human testing (also in the works).

###Data

The data currently being used is [LibriSpeech](http://www.openslr.org/12/) by Vassil Panayotov. In the future the data processing pipeline will hopefully be generalized well enough to work with any speech data. The data is fed through two pipelines, one for testing, and the other for training. This is done asynchronously.

###How to Run
####Install dependencies

1. [TensorFlow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html) (0.8)
2. sox (available on both Mac and Linux)
3. python_speech_features (pip install python_speech_features)

#####For Mac
`$ brew install sox --with-flac`
#####Ubuntu 14.04
`$ apt-get install sox`

sox is used for the dataset to convert the files from flac to wav as that was the input file requirement for the features library.

####Run Data Preperation Script

I've prepared a bash script to download (~700mb) and extract the data to the right place:

````
$ chmod +x prepare_data.sh
$ ./prepare_data.sh
````

It will remove the tar files after downloading and unzipping.

#### Change Network Parameters

All hyper parameters for the network are defined in `config.ini`. A different config file can be fed to the training program
using something like:

``$ python train.py --config_file="different_config_file.ini"``

You should ensure it follows the same format as the one I've provided.

####Running Optimizer
Once your dependencies are set up, and data is downloaded and extracted into the appropriate location, the optimizer can be started by doing:

``$ python train.py``

Dynamic RNNs are used as memory consumption on the entirely unrolled network was massive, and the model would take 30 minutes to build. Unfortunately this comes at a cost to speed, but I think in this case the tradeoff is worth it (as the model can now fit on a single GPU).

###Project Road Map

With verfication and testing performed somewhere at every step:

1. Build character-level RNN code
2. Add ctc beam search
3. Wrap acoustic model and language model into general 'Speech Recognizer'
4. Add ability for human to sample and test

Ultimately I'd like to work towards bridging this with my other project [neural-chatbot](https://github.com/inikdom/neural-chatbot)
to make an open-source natural conversational engine.

###License

MIT
