# coding=utf-8
"""
Main program to use the speech recognizer.
"""

from models.AcousticModel import AcousticModel
from models.LanguageModel import LanguageModel
from models.SpeechRecognizer import SpeechRecognizer
import tensorflow as tf
import numpy as np
import util.hyperparams as hyperparams
import util.audioprocessor as audioprocessor
import util.dataprocessor as dataprocessor
import argparse
import logging
from random import shuffle
import sys


def main():
    prog_params = parse_args()
    serializer = hyperparams.HyperParameterHandler(prog_params['config_file'])
    hyper_params = serializer.get_hyper_params()
    audio_processor = audioprocessor.AudioProcessor(hyper_params["max_input_seq_length"],
                                                    hyper_params["signal_processing"])
    # Get the input dimension for the RNN, depend on the chosen signal processing mode
    hyper_params["input_dim"] = audio_processor.feature_size

    speech_reco = SpeechRecognizer(hyper_params["language"])
    hyper_params["char_map"] = speech_reco.get_char_map()
    hyper_params["char_map_length"] = speech_reco.get_char_map_length()

    if prog_params['train_acoustic'] is True:
        if hyper_params["dataset_size_ordering"] in ['True', 'First_run_only']:
            ordered = True
        else:
            ordered = False
        train_set, test_set = speech_reco.load_acoustic_dataset(hyper_params["training_dataset_dirs"],
                                                                hyper_params["test_dataset_dirs"],
                                                                hyper_params["training_filelist_cache"],
                                                                ordered,
                                                                hyper_params["train_frac"])
        train_acoustic_rnn(train_set, test_set, hyper_params, prog_params)
    elif prog_params['train_language'] is True:
        train_set, test_set = load_language_dataset(hyper_params)
        train_language_rnn(train_set, test_set, hyper_params, prog_params)
    elif prog_params['file'] is not None:
        process_file(audio_processor, hyper_params, prog_params['file'])
    elif prog_params['record'] is True:
        record_and_write(audio_processor, hyper_params)
    elif prog_params['evaluate'] is True:
        evaluate(hyper_params)
    elif prog_params['generate_text'] is True:
        generate_text(hyper_params)


def build_language_training_rnn(sess, hyper_params, prog_params, train_set, test_set):
    model = LanguageModel(hyper_params["num_layers"], hyper_params["hidden_size"], hyper_params["batch_size"],
                          hyper_params["max_input_seq_length"], hyper_params["max_target_seq_length"],
                          hyper_params["char_map_length"])

    # Create a Dataset from the train_set and the test_set
    train_dataset = model.build_dataset(train_set, hyper_params["batch_size"], hyper_params["max_input_seq_length"])

    v_iterator = None
    if test_set is []:
        t_iterator = model.add_dataset_input(train_dataset)
        sess.run(t_iterator.initializer)
    else:
        test_dataset = model.build_dataset(test_set, hyper_params["batch_size"], hyper_params["max_input_seq_length"])

        # Build the input stream from the different datasets
        t_iterator, v_iterator = model.add_datasets_input(train_dataset, test_dataset)
        sess.run(t_iterator.initializer)
        sess.run(v_iterator.initializer)

    # Create the model
    model.create_training_rnn(hyper_params["dropout_input_keep_prob"], hyper_params["dropout_output_keep_prob"],
                              hyper_params["grad_clip"], hyper_params["learning_rate"],
                              hyper_params["lr_decay_factor"], use_iterator=True)
    model.add_tensorboard(sess, hyper_params["tensorboard_dir"], prog_params["tb_name"], prog_params["timeline"])
    model.initialize(sess)
    model.restore(sess, hyper_params["checkpoint_dir"] + "/language/")

    # Override the learning rate if given on the command line
    if prog_params["learn_rate"] is not None:
        model.set_learning_rate(sess, prog_params["learn_rate"])

    return model, t_iterator, v_iterator


def build_acoustic_training_rnn(sess, hyper_params, prog_params, train_set, test_set):
    model = AcousticModel(hyper_params["num_layers"], hyper_params["hidden_size"], hyper_params["batch_size"],
                          hyper_params["max_input_seq_length"], hyper_params["max_target_seq_length"],
                          hyper_params["input_dim"], hyper_params["batch_normalization"],
                          hyper_params["char_map_length"])

    # Create a Dataset from the train_set and the test_set
    train_dataset = model.build_dataset(train_set, hyper_params["batch_size"], hyper_params["max_input_seq_length"],
                                        hyper_params["max_target_seq_length"], hyper_params["signal_processing"],
                                        hyper_params["char_map"])

    v_iterator = None
    if test_set is []:
        t_iterator = model.add_dataset_input(train_dataset)
        sess.run(t_iterator.initializer)
    else:
        test_dataset = model.build_dataset(test_set, hyper_params["batch_size"], hyper_params["max_input_seq_length"],
                                           hyper_params["max_target_seq_length"], hyper_params["signal_processing"],
                                           hyper_params["char_map"])

        # Build the input stream from the different datasets
        t_iterator, v_iterator = model.add_datasets_input(train_dataset, test_dataset)
        sess.run(t_iterator.initializer)
        sess.run(v_iterator.initializer)

    # Create the model
    model.create_training_rnn(hyper_params["dropout_input_keep_prob"], hyper_params["dropout_output_keep_prob"],
                              hyper_params["grad_clip"], hyper_params["learning_rate"],
                              hyper_params["lr_decay_factor"], use_iterator=True)
    model.add_tensorboard(sess, hyper_params["tensorboard_dir"], prog_params["tb_name"], prog_params["timeline"])
    model.initialize(sess)
    model.restore(sess, hyper_params["checkpoint_dir"] + "/acoustic/")

    # Override the learning rate if given on the command line
    if prog_params["learn_rate"] is not None:
        model.set_learning_rate(sess, prog_params["learn_rate"])

    return model, t_iterator, v_iterator


def load_language_dataset(_hyper_params):
    # TODO : write the code...
    train_set = ["the brown lazy fox", "the red quick fox"]
    test_set = ["the white big horse", "the yellow small cat"]
    return train_set, test_set


def configure_tf_session(xla, timeline):
    # Configure tensorflow's session
    config = tf.ConfigProto()
    jit_level = 0
    if xla:
        # Turns on XLA JIT compilation.
        jit_level = tf.OptimizerOptions.ON_1
    config.graph_options.optimizer_options.global_jit_level = jit_level
    run_metadata = tf.RunMetadata()

    # Add timeline data generation options if needed
    if timeline is True:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    else:
        run_options = None
    return config, run_metadata, run_options


def train_language_rnn(train_set, test_set, hyper_params, prog_params):
    # TODO : write the code...
    config, run_metadata, run_options = configure_tf_session(prog_params["XLA"], prog_params["timeline"])

    with tf.Session(config=config) as sess:
        # Initialize the model
        build_language_training_rnn(sess, hyper_params, prog_params, train_set, test_set)

    return


def train_acoustic_rnn(train_set, test_set, hyper_params, prog_params):
    config, run_metadata, run_options = configure_tf_session(prog_params["XLA"], prog_params["timeline"])

    with tf.Session(config=config) as sess:
        # Initialize the model
        model, t_iterator, v_iterator = build_acoustic_training_rnn(sess, hyper_params,
                                                                    prog_params, train_set, test_set)

        previous_mean_error_rates = []
        current_step = epoch = 0
        while True:
            # Launch training
            mean_error_rate = 0
            for _ in range(hyper_params["steps_per_checkpoint"]):
                _step_mean_loss, step_mean_error_rate, current_step, dataset_empty =\
                    model.run_train_step(sess, hyper_params["mini_batch_size"], hyper_params["rnn_state_reset_ratio"],
                                         run_options=run_options, run_metadata=run_metadata)
                mean_error_rate += step_mean_error_rate / hyper_params["steps_per_checkpoint"]

                if dataset_empty is True:
                    epoch += 1
                    logging.info("End of epoch number : %d", epoch)
                    if (prog_params["max_epoch"] is not None) and (epoch > prog_params["max_epoch"]):
                        logging.info("Max number of epochs reached, exiting train step")
                        break
                    else:
                        # Rebuild the train dataset, shuffle it before if needed
                        if hyper_params["dataset_size_ordering"] in ['False', 'First_run_only']:
                            logging.info("Shuffling the training dataset")
                            shuffle(train_set)
                            train_dataset = model.build_dataset(train_set, hyper_params["batch_size"],
                                                                hyper_params["max_input_seq_length"],
                                                                hyper_params["max_target_seq_length"],
                                                                hyper_params["signal_processing"],
                                                                hyper_params["char_map"])
                            sess.run(t_iterator.make_initializer(train_dataset))
                        else:
                            logging.info("Reuse the same training dataset")
                            sess.run(t_iterator.initializer)

            # Save the model
            model.save(sess, hyper_params["checkpoint_dir"] + "/acoustic/")

            # Run an evaluation session
            if (current_step % hyper_params["steps_per_evaluation"] == 0) and (v_iterator is not None):
                model.run_evaluation(sess, run_options=run_options, run_metadata=run_metadata)
                sess.run(v_iterator.initializer)

            # Decay the learning rate if the model is not improving
            if mean_error_rate <= min(previous_mean_error_rates, default=sys.maxsize):
                previous_mean_error_rates.clear()
            previous_mean_error_rates.append(mean_error_rate)
            if len(previous_mean_error_rates) >= 7:
                sess.run(model.learning_rate_decay_op)
                previous_mean_error_rates.clear()
                logging.info("Model is not improving, decaying the learning rate")
                if model.learning_rate_var.eval() < 1e-7:
                    logging.info("Learning rate is too low, exiting")
                    break
                model.save(sess, hyper_params["checkpoint_dir"] + "/acoustic/")
                logging.info("Overwriting the checkpoint file with the new learning rate")

            if (prog_params["max_epoch"] is not None) and (epoch > prog_params["max_epoch"]):
                logging.info("Max number of epochs reached, exiting training session")
                break
    return


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
                              hyper_params["char_map_length"])
        model.create_forward_rnn()
        model.initialize(sess)
        model.restore(sess, hyper_params["checkpoint_dir"] + "/acoustic/")

        (a, b) = feat_vec.shape
        feat_vec = feat_vec.reshape((a, 1, b))
        predictions = model.process_input(sess, feat_vec, [original_feat_vec_length])
        transcribed_text = [dataprocessor.DataProcessor.get_labels_str(hyper_params["char_map"], prediction)
                            for prediction in predictions]
        print(transcribed_text[0])


def generate_text(hyper_params):
    with tf.Session() as sess:
        # Create model
        model = LanguageModel(hyper_params["num_layers"], hyper_params["hidden_size"], 1, 1,
                              hyper_params["max_target_seq_length"], hyper_params["char_map_length"])
        model.create_forward_rnn()
        model.initialize(sess)
        model.restore(sess, hyper_params["checkpoint_dir"] + "/language/")

        # Start with a letter
        text = "O"

        for _ in range(10):
            print(text, end="")
            # Convert to an one-hot encoded vector
            input_vec = dataprocessor.DataProcessor.get_str_to_one_hot_encoded(hyper_params["char_map"], text,
                                                                               add_eos=False)
            feat_vec = np.array(input_vec)
            (a, b) = feat_vec.shape
            feat_vec = feat_vec.reshape((a, 1, b))
            prediction = model.process_input(sess, feat_vec, [1])
            text = dataprocessor.DataProcessor.get_labels_str(hyper_params["char_map"], prediction[0])
        print(text)
        return


def evaluate(hyper_params):
    if hyper_params["test_dataset_dirs"] is None:
        logging.fatal("Setting test_dataset_dirs in config file is mandatory for evaluation mode")
        return

    # Load the test set data
    data_processor = dataprocessor.DataProcessor(hyper_params["test_dataset_dirs"])
    test_set = data_processor.get_dataset()

    logging.info("Using %d size of test set", len(test_set))

    if len(test_set) == 0:
        logging.fatal("No files in test set during an evaluation mode")
        return

    with tf.Session() as sess:
        # create model
        model = AcousticModel(hyper_params["num_layers"], hyper_params["hidden_size"], hyper_params["batch_size"],
                              hyper_params["max_input_seq_length"], hyper_params["max_target_seq_length"],
                              hyper_params["input_dim"], hyper_params["batch_normalization"],
                              hyper_params["char_map_length"])

        model.create_forward_rnn()
        model.initialize(sess)
        model.restore(sess, hyper_params["checkpoint_dir"] + "/acoustic/")

        wer, cer = model.evaluate_full(sess, test_set, hyper_params["max_input_seq_length"],
                                       hyper_params["signal_processing"], hyper_params["char_map"])
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
                              hyper_params["char_map_length"])

        model.create_forward_rnn()
        model.initialize(sess)
        model.restore(sess, hyper_params["checkpoint_dir"] + "/acoustic/")

        # Create stream of listening
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=_SR, input=True, frames_per_buffer=_CHUNK)
        print("NOW RECORDING...")

        while True:
            data = stream.read(_CHUNK)
            data = np.fromstring(data)
            feat_vec, original_feat_vec_length = audio_processor.process_signal(data, _SR)
            (a, b) = feat_vec.shape
            feat_vec = feat_vec.reshape((a, 1, b))
            predictions = model.process_input(sess, feat_vec, [original_feat_vec_length])
            result = [dataprocessor.DataProcessor.get_labels_str(hyper_params["char_map"], prediction)
                      for prediction in predictions]
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
    parser.set_defaults(XLA=False)
    parser.add_argument('--XLA', dest='XLA', action='store_true', help='Activate XLA mode in tensorflow')

    group = parser.add_mutually_exclusive_group(required=True)
    group.set_defaults(train_acoustic=False)
    group.set_defaults(train_language=False)
    group.set_defaults(file=None)
    group.set_defaults(record=False)
    group.set_defaults(evaluate=False)
    group.add_argument('--train_acoustic', dest='train_acoustic', action='store_true',
                       help='Train the acoustic network')
    group.add_argument('--train_language', dest='train_language', action='store_true',
                       help='Train the language network')
    group.add_argument('--file', type=str, help='Path to a wav file to process')
    group.add_argument('--record', dest='record', action='store_true', help='Record and write result on the fly')
    group.add_argument('--evaluate', dest='evaluate', action='store_true', help='Evaluate WER against the test_set')
    group.add_argument('--generate_text', dest='generate_text', action='store_true', help='Generate text from the '
                                                                                          'language model')

    args = parser.parse_args()
    prog_params = {'config_file': args.config, 'tb_name': args.tb_name, 'max_epoch': args.max_epoch,
                   'learn_rate': args.learn_rate, 'timeline': args.timeline, 'train_acoustic': args.train_acoustic,
                   'train_language': args.train_language, 'file': args.file, 'record': args.record,
                   'evaluate': args.evaluate, 'generate_text': args.generate_text, 'XLA': args.XLA}
    return prog_params


if __name__ == "__main__":
    main()
