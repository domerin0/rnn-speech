# coding=utf-8
"""
Main program to use the speech recognizer.
"""

from models.AcousticModel import AcousticModel
import tensorflow as tf
import numpy as np
import util.hyperparams as hyperparams
import util.audioprocessor as audioprocessor
import util.dataprocessor as dataprocessor
import argparse
from math import floor
import logging
from random import shuffle


def main():
    prog_params = parse_args()
    serializer = hyperparams.HyperParameterHandler(prog_params['config_file'])
    hyper_params = serializer.get_hyper_params()
    audio_processor = audioprocessor.AudioProcessor(hyper_params["max_input_seq_length"],
                                                    hyper_params["signal_processing"])
    # Get the input dimension for the RNN, depend on the chosen signal processing mode
    hyper_params["input_dim"] = audio_processor.feature_size

    if prog_params['train'] is True:
        train_set, test_set = load_training_dataset(audio_processor, hyper_params)
        train_rnn(train_set, test_set, hyper_params, prog_params)
    elif prog_params['file'] is not None:
        process_file(audio_processor, hyper_params, prog_params['file'])
    elif prog_params['record'] is True:
        record_and_write(audio_processor, hyper_params)
    elif prog_params['evaluate'] is True:
        evaluate(audio_processor, hyper_params)


def build_training_rnn(sess, hyper_params, prog_params, overriden_max_input_seq_length=None):
    if overriden_max_input_seq_length is None:
        overriden_max_input_seq_length = hyper_params["max_input_seq_length"]
    model = AcousticModel(hyper_params["num_layers"], hyper_params["hidden_size"], hyper_params["batch_size"],
                          overriden_max_input_seq_length, hyper_params["max_target_seq_length"],
                          hyper_params["input_dim"], hyper_params["batch_normalization"],
                          language=hyper_params["language"])

    model.create_training_rnn(hyper_params["dropout_input_keep_prob"], hyper_params["dropout_output_keep_prob"],
                              hyper_params["grad_clip"], hyper_params["learning_rate"],
                              hyper_params["lr_decay_factor"])

    model.add_tensorboard(sess, hyper_params["tensorboard_dir"], prog_params["tb_name"], prog_params["timeline"])
    model.initialize(sess)
    model.restore(sess, hyper_params["checkpoint_dir"])
    return model


def load_training_dataset(audio_processor, hyper_params):
    # Load the train set data
    data_processor = dataprocessor.DataProcessor(hyper_params["training_dataset_dirs"], audio_processor,
                                                 file_cache=hyper_params["training_filelist_cache"],
                                                 size_ordering=hyper_params["dataset_size_ordering"])
    train_set = data_processor.run()
    if hyper_params["test_dataset_dirs"] is not None:
        # Load the test set data
        data_processor = dataprocessor.DataProcessor(hyper_params["test_dataset_dirs"], audio_processor)
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
    return train_set, test_set


def train_rnn(train_set, test_set, hyper_params, prog_params):
    # Configure tensorflow's session
    config = tf.ConfigProto()
    jit_level = 0
    if prog_params["XLA"]:
        # Turns on XLA JIT compilation.
        jit_level = tf.OptimizerOptions.ON_1
    config.graph_options.optimizer_options.global_jit_level = jit_level
    run_metadata = tf.RunMetadata()

    # Add timeline data generation options if needed
    if prog_params["timeline"] is True:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    else:
        run_options = None

    full_run_size = hyper_params["batch_size"] * hyper_params["mini_batch_size"]
    end_pos = None
    previous_mean_error_rate = None
    step_num = 0
    epoch = 1
    while True:
        # Select training data
        if end_pos is None:
            start_pos = prog_params["start_from"]
            if start_pos > len(train_set):
                raise ValueError("Invalid program parameter 'start_from' : higher than training dataset size")
        else:
            start_pos = end_pos
        end_pos = (start_pos + (full_run_size * hyper_params["steps_per_checkpoint"])) % len(train_set)
        # Check if end of training set is reached
        if end_pos > start_pos:
            session_set = train_set[start_pos:end_pos]
        else:
            session_set = train_set[start_pos:]
            session_set += train_set[:end_pos]
            epoch += 1
            if hyper_params["dataset_size_ordering"] in ['False', 'First_run_only']:
                logging.info("New epoch {0}: shuffling training dataset for next time".format(epoch))
                shuffle(train_set)

        # Find max_input_seq_length for this training data
        local_input_seq_length = max(session_set, key=lambda x: x[2])[2] + 10
        local_input_seq_length = min(local_input_seq_length, hyper_params["max_input_seq_length"])
        logging.info("Start a session with local_input_seq_length = %d", local_input_seq_length)

        # Run a training session
        with tf.Session(config=config) as sess:
            # Create model with the local max_input_seq_length for that part of the dataset
            model = build_training_rnn(sess, hyper_params, prog_params,
                                       overriden_max_input_seq_length=local_input_seq_length)

            # Create a local audio_processor set for the local max_input_seq_length
            local_audio_processor = audioprocessor.AudioProcessor(local_input_seq_length,
                                                                  hyper_params["signal_processing"])
            # Run training
            _, _ = model.fit(sess, local_audio_processor, session_set, hyper_params["mini_batch_size"],
                             rnn_state_reset_ratio=hyper_params["rnn_state_reset_ratio"],
                             run_options=run_options, run_metadata=run_metadata)
            model.save(sess, hyper_params["checkpoint_dir"])
            step_num += hyper_params["steps_per_checkpoint"]

        tf.reset_default_graph()

        # Run an evaluation session
        if step_num % hyper_params["steps_per_evaluation"] == 0:
            with tf.Session(config=config) as sess:
                # Find max_input_seq_length for this evaluation data
                local_input_seq_length = max(test_set, key=lambda x: x[2])[2] + 10
                local_input_seq_length = min(local_input_seq_length, hyper_params["max_input_seq_length"])
                logging.info("Start a session with local_input_seq_length = %d", local_input_seq_length)

                # create model
                model = build_training_rnn(sess, hyper_params, prog_params,
                                           overriden_max_input_seq_length=local_input_seq_length)

                # Create a local audio_processor set for the local max_input_seq_length
                local_audio_processor = audioprocessor.AudioProcessor(local_input_seq_length,
                                                                      hyper_params["signal_processing"])

                # Evaluate
                mean_loss, mean_error_rate, _ = model.evaluate_basic(sess, test_set, local_audio_processor,
                                                                     run_options=run_options, run_metadata=run_metadata)

                # Decay the learning rate if the model is not improving
                if previous_mean_error_rate is not None:
                    if mean_error_rate > previous_mean_error_rate:
                        sess.run(model.learning_rate_decay_op)
                        logging.info("Model is not improving, decaying the learning rate")
                        if model.learning_rate_var.eval() < 1e-7:
                            logging.info("Learning rate is too low, exiting")
                            break
                        model.save(sess, hyper_params["checkpoint_dir"])
                        logging.info("Overwriting the checkpoint file with the new learning rate")
                previous_mean_error_rate = mean_error_rate

            tf.reset_default_graph()

        if (prog_params["max_epoch"] is not None) and (epoch > prog_params["max_epoch"]):
            break


def process_file(audio_processor, hyper_params, file):
    feat_vec, original_feat_vec_length = audio_processor.process_audio_file(file)
    if original_feat_vec_length > hyper_params["max_input_seq_length"]:
        logging.warning("File too long")
        return

    with tf.Session() as sess:
        # create model
        model = AcousticModel(hyper_params["num_layers"], hyper_params["hidden_size"], 1,
                              hyper_params["max_input_seq_length"], hyper_params["max_target_seq_length"],
                              hyper_params["input_dim"], hyper_params["batch_normalization"],
                              language=hyper_params["language"])
        model.create_forward_rnn(with_input_queue=False)
        model.initialize(sess)
        model.restore(sess, hyper_params["checkpoint_dir"])

        (a, b) = feat_vec.shape
        feat_vec = feat_vec.reshape((a, 1, b))
        transcribed_text = model.process_input(sess, feat_vec, [original_feat_vec_length])
        print(transcribed_text[0])


def evaluate(audio_processor, hyper_params):
    if hyper_params["test_dataset_dirs"] is None:
        logging.fatal("Setting test_dataset_dirs in config file is mandatory for evaluation mode")
        return

    # Load the test set data
    data_processor = dataprocessor.DataProcessor(hyper_params["test_dataset_dirs"], audio_processor)
    test_set = data_processor.run()

    logging.info("Using %d size of test set", len(test_set))

    if len(test_set) == 0:
        logging.fatal("No files in test set during an evaluation mode")
        return

    with tf.Session() as sess:
        # create model
        model = AcousticModel(hyper_params["num_layers"], hyper_params["hidden_size"], hyper_params["batch_size"],
                              hyper_params["max_input_seq_length"], hyper_params["max_target_seq_length"],
                              hyper_params["input_dim"], hyper_params["batch_normalization"],
                              language=hyper_params["language"])

        model.create_forward_rnn()
        model.initialize(sess)
        model.restore(sess, hyper_params["checkpoint_dir"])

        wer, cer = model.evaluate_full(sess, test_set, audio_processor)
        print("Resulting WER : {0:.3g} %".format(wer))
        print("Resulting CER : {0:.3g} %".format(cer))
        return


def record_and_write(audio_processor, hyper_params):
    import pyaudio
    _CHUNK = hyper_params["max_input_seq_length"]
    _SR = 22050
    p = pyaudio.PyAudio()

    with tf.Session() as sess:
        # create model
        model = AcousticModel(hyper_params["num_layers"], hyper_params["hidden_size"], 1,
                              hyper_params["max_input_seq_length"], hyper_params["max_target_seq_length"],
                              hyper_params["input_dim"], hyper_params["batch_normalization"],
                              language=hyper_params["language"])

        model.create_forward_rnn()
        model.initialize(sess)
        model.restore(sess, hyper_params["checkpoint_dir"])

        # Create stream of listening
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=_SR, input=True, frames_per_buffer=_CHUNK)
        print("NOW RECORDING...")

        while True:
            data = stream.read(_CHUNK)
            data = np.fromstring(data)
            feat_vec, original_feat_vec_length = audio_processor.extraction_f(data, _SR)
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
    parser.add_argument('--start_from', type=int, default=0,
                        help='Position in the train set to start training from (default : 0)')
    parser.add_argument('--max_epoch', type=int, default=None,
                        help='Max epoch to train (no limitation if not provided)')
    parser.add_argument('--learn_rate', type=float, default=None,
                        help='Force learning rate to start from this value (overriding checkpoint value)')
    parser.set_defaults(timeline=False)
    parser.add_argument('--timeline', dest='timeline', action='store_true',
                        help='Generate a json file with the timeline (a tensorboard directory'
                             'must be provided in config file)')
    parser.set_defaults(XLA=False)
    parser.add_argument('--XLA', dest='XLA', action='store_true', help='Activate XLA mode in tensorflow')

    group = parser.add_mutually_exclusive_group(required=True)
    group.set_defaults(train=False)
    group.set_defaults(file=None)
    group.set_defaults(record=False)
    group.set_defaults(evaluate=False)
    group.add_argument('--train', dest='train', action='store_true', help='Train the network')
    group.add_argument('--file', type=str, help='Path to a wav file to process')
    group.add_argument('--record', dest='record', action='store_true', help='Record and write result on the fly')
    group.add_argument('--evaluate', dest='evaluate', action='store_true', help='Evaluate WER against the test_set')

    args = parser.parse_args()
    prog_params = {'config_file': args.config, 'tb_name': args.tb_name, 'start_from': args.start_from,
                   'max_epoch': args.max_epoch, 'learn_rate': args.learn_rate, 'timeline': args.timeline,
                   'train': args.train, 'file': args.file, 'record': args.record, 'evaluate': args.evaluate,
                   'XLA': args.XLA}
    return prog_params


if __name__ == "__main__":
    main()
