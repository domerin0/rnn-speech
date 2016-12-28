# English pre-trained model

This directory contain a trained english acoustic model (77.000 steps of 3 files batches).
This is still work in progress, far from perfect, but it give some insight on the current state of the software

### Dataset

The dataset was built using [Shtooka](http://shtooka.net/) and [LibriSpeech](http://www.openslr.org/12/) datasets

Training set contains :
* LibriSpeech's train-clean-100
* LibriSpeech's train-clean-360
* Shtooka's eng-balm-emmanuel_flac.tar
* Shtooka's eng-balm-judith_flac.tar
* Shtooka's eng-balm-verbs_flac.tar
* Shtooka's eng-wcp-us_flac.tar
* Shtooka's eng-wims-mary_flac.tar

Test set contains :
* LibriSpeech's test-clean

This version of the software use **size ordering** of the training set in order to improve performance and accuracy. 

### How to try it
Make sure to set those parameters in your config file :

    [acoustic_network_params]
    num_layers : 2
    hidden_size : 768
    dropout : 0.5
    batch_size : 3
    learning_rate : 3e-4
    lr_decay_factor : 0.90
    grad_clip : 5
        
    [general]
    use_config_file_if_checkpoint_exists : True
    steps_per_checkpoint : 1000
    checkpoint_dir : trained_models/acoustic_model/english_Shtooka_and_Libri
        
    [training]
    max_input_seq_length : 800
    max_target_seq_length : 300


Run the model on a "less than 8 seconds long" wav file of your choice

    python3 stt.py --file data/LibriSpeech/dev-clean/2086/149220/2086-149220-0007.flac

On this example file from Librispeech dev set that the model never trained on you will obtain :

    it now contain oly shonto clare is to waves and the tolitary chickn

the original text being :

    it now contained only chanticleer his two wives and a solitary chicken

### Reproduce the learning phase
Put the training data directories in a "train" directory and set it in config file

    training_dataset_dirs : data/Shtooka/train, data/LibriSpeech/train

Put the test data in another directory and set it in config file

    test_dataset_dirs : data/LibriSpeech/test

Launch training and wait...

    python3 stt.py --train --tb_name libri_shoota_size_ordered


### Training graphs

The learning rate was initialized at 3e-4 and automatically reduced at 2.7e-4 at the beginning of the learning.
![Learning rate](learning_rate.png)


The error rate below is on the test set on which the rnn never train.
![Error rate on test set](error_rate_test.png)


The loss is dropping, except for a bump at 47.000 steps where the training was interrupted and re-launched.
![Loss](loss.png)

