# rnn-speech
Character level speech recognizer using ctc loss with deep rnns in TensorFlow.

###About

This is an ongoing project, working towards an implementation of the charater-level ISR detailed in the [paper](http://arxiv.org/pdf/1601.06581v2.pdf) by Kyuyeon Hwang and Wonyong Sung. It works at the character level using 1 deep rnn trained with ctc loss for the acoustic model, and one deep rnn trained for a character-level language model. The acoustic model reads in log mel frequency filterbank feature vectors with energy, delta and delta-delta values (123-dim inputs).

The audio signal processing is done using jameslyons' [python_speech_features](https://github.com/jameslyons/python_speech_features), and this [MFCC tutorial](http://www.practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/).

Currently only the acoustic model has been completed and it still lack a good trained example.
One pre-trained example is available [here](trained_models/acoustic_model/english_Shtooka/README.md) and can be tried on any file (your own recorded voice for example).

The character-level language model is still in the works.

###Data

The datasets currently supported are :
* [LibriSpeech](http://www.openslr.org/12/) by Vassil Panayotov
* [Shtooka](http://shtooka.net/)
* [Vystadial 2013](http://hdl.handle.net/11858/00-097C-0000-0023-4670-6)
* [TED-LIUM](http://www-lium.univ-lemans.fr/en/content/ted-lium-corpus)

The data is fed through two pipelines, one for testing, and the other for training. This can be done asynchronously or the resulting input vector can also be saved to avoid re-processing.

###How to Run
####Install dependencies

1. [TensorFlow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html) (>= 0.8)
1. sox (available on both Mac and Linux)
1. python_speech_features (pip3 install python_speech_features)
1. h5py (pip3 install h5py)
1. tk (sudo apt-get install python3-tk)

#####For Mac
`$ brew install sox --with-flac`
#####Ubuntu 14.04
`$ apt-get install sox`

sox is used for the dataset to convert the files from flac to wav as that was the input file requirement for the features library.

####Run data preparation Script

I've prepared a bash script to download LibriSpeech (~700mb) and extract the data to the right place :

````
$ chmod +x prepare_data.sh
$ ./prepare_data.sh
````

It will remove the tar files after downloading and unzipping.

#### Change Network Parameters

All hyper parameters for the network are defined in `config.ini`. A different config file can be fed to the training program using something like:

``$ python stt.py --config_file="different_config_file.ini"``

You should ensure it follows the same format as the one provided.

####Running Optimizer
Once your dependencies are set up, and data is downloaded and extracted into the appropriate location, the optimizer can be started by doing :

``$ python stt.py --train``

Dynamic RNNs are used as memory consumption on the entirely unrolled network was massive, and the model would take 30 minutes to build. Unfortunately this comes at a cost to speed, but I think in this case the tradeoff is worth it (as the model can now fit on a single GPU).

####Running the network
You can also use a trained network to process a wav file

``$ python stt.py --file "path_to_file.wav"``

The result will be printed on standard input. At this time only the acoustic model will process so the result can be weird.  

###Project Road Map

With verification and testing performed somewhere at every step:

1. ~~Build character-level RNN code~~
2. Add ctc beam search
3. Wrap acoustic model and language model into general 'Speech Recognizer'
4. Add ability for human to sample and test

Ultimately I'd like to work towards bridging this with my other project [neural-chatbot](https://github.com/inikdom/neural-chatbot)
to make an open-source natural conversational engine.

###License

MIT


###References
#### LibriSpeech
````
"LibriSpeech: an ASR corpus based on public domain audio books", Vassil Panayotov, Guoguo Chen, Daniel Povey and Sanjeev Khudanpur, ICASSP 2015
````

#### Shtooka
````
http://shtooka.net
````

####Vystadial 2013
````
Korvas, Matěj; Plátek, Ondřej; Dušek, Ondřej; Žilka, Lukáš and Jurčíček, Filip, 2014, Vystadial 2013 – Czech data,
LINDAT/CLARIN digital library at Institute of Formal and Applied Linguistics, Charles University in Prague,
http://hdl.handle.net/11858/00-097C-0000-0023-4670-6.
````

#### TED-LIUM Corpus
````
A. Rousseau, P. Deléglise, and Y. Estève, "Enhancing the TED-LIUM Corpus with Selected Data for Language Modeling and More TED Talks",
in Proceedings of the Ninth International Conference on Language Resources and Evaluation (LREC’14), May 2014.
````
