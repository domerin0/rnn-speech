# English pre-trained model

This directory contain a trained english acoustic model (53.000 steps of 10 files batches).

__Warning :__ This model use 3 layers, default is 2 in config file

Results on LibriSpeech's test-clean evaluation set :
* __CER : 21,3 %__
* __WER : 56,2 %__


### Dataset

The dataset was built using [Shtooka](http://shtooka.net/), [LibriSpeech](http://www.openslr.org/12/) and 
[TED-LIUM](http://www-lium.univ-lemans.fr/en/content/ted-lium-corpus) datasets

Training set contains :
* LibriSpeech's train-clean-100
* LibriSpeech's train-clean-360
* LibriSpeech's train-other-500
* Shtooka's eng-balm-emmanuel_flac.tar
* Shtooka's eng-balm-judith_flac.tar
* Shtooka's eng-balm-verbs_flac.tar
* Shtooka's eng-wcp-us_flac.tar
* Shtooka's eng-wims-mary_flac.tar
* TED-LIUM's release 2

Test set contains :
* LibriSpeech's test-clean

### How to try it
First check that you have downloaded the lfs managed files. Execute this command in project's root dir :

    $ git lfs pull

Then make sure to set those parameters in your config file :

    [acoustic_network_params]
    num_layers : 3
    hidden_size : 768
    dropout_input_keep_prob : 0.8
    dropout_output_keep_prob : 0.5
    batch_size : 10
    learning_rate : 0.0003
    lr_decay_factor : 0.33
    grad_clip : 5
    signal_processing : fbank
        
    [general]
    use_config_file_if_checkpoint_exists : True
    steps_per_checkpoint : 1000
    checkpoint_dir : trained_models/acoustic_model/english
        
    [training]
    max_input_seq_length : 1800
    max_target_seq_length : 600
    batch_normalization : False
    dataset_size_ordering : False


Run the model on a "less than 15 seconds long" wav file of your choice

    $ python3 stt.py --file data/LibriSpeech/dev-clean/2086/149220/2086-149220-0007.flac

On this example file from Librispeech dev set that the model never trained on you will obtain :

    it now contained only shon to clar his two wides and a selitary chicken

the original text being :

    it now contained only chanticleer his two wives and a solitary chicken

### Reproduce the learning phase
Put the training data directories in a "train" directory and set it in config file

    training_dataset_dirs : data/Shtooka/train, data/LibriSpeech/train, data/TEDLIUM_release2/train

Put the test data in another directory and set it in config file

    test_dataset_dirs : data/LibriSpeech/test

Launch training and wait...

    $ python3 stt.py --train --tb_name libri_shoota_TEDLIUM


### Training graphs

The learning rate was initialized at 0.0003 and was lowered when result on the test set did not improve.
A bug prevented it to drop from steps 6.000 to 34.000. Training was stopped and relaunched at 34.000 after the
bug was fixed.
![Learning rate](learning_rate.png)


The error rate below is on the test set on which the rnn never train.
Even with 3 layers the CER does not drop below 20 %
![Error rate on test set](error_rate_test.png)


The loss is dropping as expected.
![Loss](loss.png)

