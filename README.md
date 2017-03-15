# rnn-speech
Character level speech recognizer using ctc loss with deep rnns in TensorFlow.

### About

This is an ongoing project, working towards an implementation of the charater-level ISR detailed in the
[paper](http://arxiv.org/pdf/1601.06581v2.pdf) by Kyuyeon Hwang and Wonyong Sung. It works at the character level
using 1 deep rnn trained with ctc loss for the acoustic model, and one deep rnn trained for a character-level language
model. The acoustic model can read in either mel frequency cepstral coefficient, or mel filterbank with delta and
double delta feature vectors (40 or 120 dim inputs respectively).

The audio signal processing is done using [librosa](https://github.com/librosa/librosa).

Currently only the acoustic model has been completed.
One pre-trained example is available [here](trained_models/acoustic_model/english) and can be tried
on any file (your own recorded voice for example).

Results on LibriSpeech's test-clean evaluation set for the pre-trained model is :
* __CER : 19,5 %__
* __WER : 52 %__

It lacks the character-level language model which is still in the works.

### Data

The datasets currently supported are :
* [LibriSpeech](http://www.openslr.org/12/) by Vassil Panayotov
* [Shtooka](http://shtooka.net/)
* [Vystadial 2013](http://hdl.handle.net/11858/00-097C-0000-0023-4670-6)
* [TED-LIUM](http://www-lium.univ-lemans.fr/en/content/ted-lium-corpus)

The data is fed through two pipelines, one for testing, and the other for training.

### How to Run
#### Cloning the repository
If you intend to use a pre-trained model you should clone the repository with the
[lfs plugin](https://git-lfs.github.com/)

    $ git lfs clone https://github.com/inikdom/rnn-speech.git

If you have already cloned the repository without lfs, you can download the missing files with :

    $ git lfs pull


#### Install dependencies
##### Required

1. [TensorFlow](https://www.tensorflow.org) (>= 1.0)
1. [librosa](https://github.com/librosa/librosa)

Install required dependencies by running :

    $ pip3 install -r requirements.txt

GPU support is not mandatory but strongly recommended if you intend to train the RNN.
Replace tensorflow by tensorflow-gpu in requirements.txt in order to install the GPU 
version of TensorFlow.

##### Optional
1. sox (for live transcript only, install with `sudo apt-get install sox` or `brew install sox --with-flac`)
1. libcupti (for timeline only, install with : `sudo apt-get install libcupti-dev`)
1. pyaudio (for live transcript only, install with : `sudo apt-get install python3-pyaudio`)


#### Run data preparation Script

I've prepared a bash script to download LibriSpeech (~700mb) and extract the data to the right place :

    $ chmod +x prepare_data.sh
    $ ./prepare_data.sh

It will remove the tar files after downloading and unzipping.

#### Change Network Parameters

All hyper parameters for the network are defined in `config.ini`. A different config file can be fed to the training
program using something like:

    $ python stt.py --config_file="different_config_file.ini"

You should ensure it follows the same format as the one provided.

#### Running Optimizer
Once your dependencies are set up, and data is downloaded and extracted into the appropriate location,
the optimizer can be started by doing :

    $ python stt.py --train

Dynamic RNNs are used as memory consumption on the entirely unrolled network was massive, and the model would take
30 minutes to build. Unfortunately this comes at a cost to speed, but I think in this case the tradeoff is worth it
(as the model can now fit on a single GPU).

#### Running the network
You can also use a trained network to process a wav file

    $ python stt.py --file "path_to_file.wav"

The result will be printed on standard input.

#### Evaluating the network
You can evaluate a trained network on a evaluation test set (config.ini file's _test_dataset_dirs_ parameter)

    $ python stt.py --evaluate

The resulting CER (character error rate) and WER (word error rate) will be printed on standard input.

#### Analysing performance
You can add the `--timeline` option in order to produce a timeline file and see how everything is going.

The resulting file will be overridden at each step. It can be opened with Chrome, opening `chrome://tracing/` and
loading the file.

### Project Road Map

With verification and testing performed somewhere at every step:

1. ~~Build character-level RNN code~~
2. Add ctc beam search
3. Wrap acoustic model and language model into general 'Speech Recognizer'
4. Add ability for human to sample and test

Ultimately I'd like to work towards bridging this with my other project
[neural-chatbot](https://github.com/inikdom/neural-chatbot) to make an open-source natural conversational engine.

### License

MIT


### References
#### LibriSpeech
````
"LibriSpeech: an ASR corpus based on public domain audio books", Vassil Panayotov, Guoguo Chen, Daniel Povey andSanjeev Khudanpur, ICASSP 2015
````

#### Shtooka
````
http://shtooka.net
````

#### Vystadial 2013
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
