# coding=utf-8
"""
Main program to use the speech recognizer.
"""

from models.AcousticModel import AcousticModel
import tensorflow as tf
import util.hyperparams as hyperparams
import util.audioprocessor as audioprocessor
import util.dataprocessor as dataprocessor
import argparse
from math import floor
import logging


def main():
    prog_params = parse_args()
    serializer = hyperparams.HyperParameterHandler(prog_params['config_file'])
    hyper_params = serializer.getHyperParams()
    audio_processor = audioprocessor.AudioProcessor(hyper_params["max_input_seq_length"],
                                                    hyper_params["signal_processing"])
    # Get the input dimension for the RNN, depend on the chosen signal processing mode
    hyper_params["input_dim"] = audio_processor.feature_size

    if prog_params['train'] is True:
        train_rnn(audio_processor, hyper_params, prog_params)
    elif prog_params['file'] is not None:
        process_file(audio_processor, hyper_params, prog_params['file'])
    elif prog_params['record'] is True:
        record_and_write(audio_processor, hyper_params)


def train_rnn(audio_processor, hyper_params, prog_params):
    # Load the train set data
    data_processor = dataprocessor.DataProcessor(hyper_params["training_dataset_dirs"],
                                                 audio_processor, size_ordering=True)
    train_set = data_processor.run()
    if hyper_params["test_dataset_dirs"] is not None:
        # Load the test set data
        data_processor = dataprocessor.DataProcessor(hyper_params["test_dataset_dirs"],
                                                     audio_processor, size_ordering=True)
        test_set = data_processor.run()
    elif hyper_params["train_frac"] is not None:
        # Or use a fraction of the train set for the test set
        num_train = max(1, int(floor(hyper_params["train_frac"] * len(train_set))))
        test_set = train_set[num_train:]
        train_set = train_set[:num_train]
    else:
        # Or use no test set
        test_set = []

    logging.info("Using %d files in train set", len(train_set))
    logging.info("Using %d size of test set", len(test_set))

    with tf.Session() as sess:
        # create model
        model = create_acoustic_model(sess, hyper_params, hyper_params["batch_size"],
                                      forward_only=False, tensorboard_dir=hyper_params["tensorboard_dir"],
                                      tb_run_name=prog_params["tb_name"], timeline_enabled=prog_params["timeline"])
        # Override the learning rate if given on the command line
        if prog_params["learn_rate"] is not None:
            assign_op = model.learning_rate.assign(prog_params["learn_rate"])
            sess.run(assign_op)

        logging.info("Start training...")
        model.train(sess, audio_processor, test_set, train_set, hyper_params["steps_per_checkpoint"],
                    hyper_params["checkpoint_dir"], max_epoch=prog_params["max_epoch"])


def process_file(audio_processor, hyper_params, file):
    feat_vec, original_feat_vec_length = audio_processor.process_audio_file(file)
    if original_feat_vec_length > hyper_params["max_input_seq_length"]:
        logging.warning("File too long")
        return

    with tf.Session() as sess:
        # create model
        model = create_acoustic_model(sess, hyper_params, 1, forward_only=True, tensorboard_dir=None,
                                      tb_run_name=None, timeline_enabled=False)

        (a, b) = feat_vec.shape
        feat_vec = feat_vec.reshape((a, 1, b))
        transcribed_text = model.process_input(sess, feat_vec, [original_feat_vec_length])
        print(transcribed_text)


def create_acoustic_model(session, hyper_params, batch_size, forward_only=True, tensorboard_dir=None,
                          tb_run_name=None, timeline_enabled=False):
    num_labels = 31
    logging.info("Building model... (this takes a while)")
    model = AcousticModel(session, num_labels, hyper_params["num_layers"], hyper_params["hidden_size"],
                          hyper_params["dropout"], batch_size, hyper_params["learning_rate"],
                          hyper_params["lr_decay_factor"], hyper_params["grad_clip"],
                          hyper_params["max_input_seq_length"], hyper_params["max_target_seq_length"],
                          hyper_params["input_dim"], forward_only=forward_only, tensorboard_dir=tensorboard_dir,
                          tb_run_name=tb_run_name, timeline_enabled=timeline_enabled)
    ckpt = tf.train.get_checkpoint_state(hyper_params["checkpoint_dir"])
    # Initialize variables
    session.run(tf.global_variables_initializer())
    # Restore from checkpoint (will overwrite variables)
    if ckpt:
        logging.info("Reading model parameters from %s", ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
    return model


def record_and_write(audio_processor, hyper_params):
    import pyaudio
    import numpy as np
    _CHUNK = hyper_params["max_input_seq_length"]
    _SR = 22050
    p = pyaudio.PyAudio()

    with tf.Session() as sess:
        # create model
        model = create_acoustic_model(sess, hyper_params, 1, forward_only=True, tensorboard_dir=None,
                                      tb_run_name=None, timeline_enabled=False)
        # Create stream of listening
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=_SR, input=True, frames_per_buffer=_CHUNK)
        print("NOW RECORDING...")

        while True:
            data = stream.read(_CHUNK)
            data = np.fromstring(data)
            feat_vec, original_feat_vec_length = audio_processor.extract_mfcc(data, _SR)
            (a, b) = feat_vec.shape
            feat_vec = feat_vec.reshape((a, 1, b))
            result = model.process_input(sess, feat_vec, [original_feat_vec_length])
            print(result, end="")


def parse_args():
    """
    Parses the command line input.

    """
    _DEFAULT_CONFIG_FILE = 'config.ini'

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=_DEFAULT_CONFIG_FILE,
                        help='Path to configuration file with hyper-parameters.')
    parser.add_argument('--tb_name', type=str, default=None,
                        help='Tensorboard path name for the run (allow multiples run with the same output path)')
    parser.add_argument('--max_epoch', type=int, default=None,
                        help='Max epoch to train (no limitation if not provided)')
    parser.add_argument('--learn_rate', type=float, default=None,
                        help='Force learning rate to start from this value (overriding checkpoint value)')
    parser.set_defaults(timeline=False)
    parser.add_argument('--timeline', dest='timeline', action='store_true',
                        help='Generate a json file with the timeline (a tensorboard directory'
                             'must be provided in config file)')

    group = parser.add_mutually_exclusive_group(required=True)
    group.set_defaults(train=False)
    group.set_defaults(file=None)
    group.set_defaults(record=False)
    group.add_argument('--train', dest='train', action='store_true', help='Train the network')
    group.add_argument('--file', type=str, help='Path to a wav file to process')
    group.add_argument('--record', dest='record', action='store_true', help='Record and write result on the fly')

    args = parser.parse_args()
    prog_params = {'config_file': args.config, 'tb_name': args.tb_name, 'max_epoch': args.max_epoch,
                   'learn_rate': args.learn_rate, 'timeline': args.timeline, 'train': args.train,
                   'file': args.file, 'record': args.record}
    return prog_params


if __name__ == "__main__":
    main()
