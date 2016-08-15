'''
Main program to use the speech recognizer.
'''

from tensorflow.python.platform import gfile
from models.AcousticModel import AcousticModel
import tensorflow as tf
import util.hyperparams as hyperparams
import util.audioprocessor as audioprocessor
import util.dataprocessor as dataprocessor
import argparse
from math import floor


def main():
    prog_params = parse_args()
    serializer = hyperparams.HyperParameterHandler(prog_params['config_file'])
    hyper_params = serializer.getHyperParams()
    audio_processor = audioprocessor.AudioProcessor(hyper_params["max_input_seq_length"],
                                                    hyper_params["load_save_input_vec"])

    if prog_params['train'] is True:
        train_rnn(hyper_params)
    else:
        process_file(audio_processor, hyper_params, prog_params['file'])


def train_rnn(hyper_params):
    data_processor = dataprocessor.DataProcessor(hyper_params["training_dataset_dir"])
    text_audio_pairs = data_processor.run()
    num_train = int(floor(hyper_params["train_frac"] * len(text_audio_pairs)))
    train_set = text_audio_pairs[:num_train]
    test_set = text_audio_pairs[num_train:]
    print("Using {0} size of test set".format(len(test_set)))

    with tf.Session() as sess:
        # create model
        print("Building model... (this takes a while)")
        model = createAcousticModel(sess, hyper_params, hyper_params["batch_size"], False)
        print("Setting up audio processor...")
        model.initializeAudioProcessor(hyper_params["max_input_seq_length"], hyper_params["load_save_input_vec"])
        print("Start training...")
        model.train(sess, test_set, train_set, hyper_params["steps_per_checkpoint"], hyper_params["checkpoint_dir"])


def process_file(audio_processor, hyper_params, file):
    feat_vec, original_feat_vec_length = audio_processor.processFLACAudio(file)
    if original_feat_vec_length > hyper_params["max_input_seq_length"]:
        print("File too long")
        return

    with tf.Session() as sess:
        # create model
        print("Building model... (this takes a while)")
        model = createAcousticModel(sess, hyper_params, 1, True)

        (a, b) = feat_vec.shape
        feat_vec = feat_vec.reshape((a, 1, b))
        logit = model.process_input(sess, feat_vec, [original_feat_vec_length])
        logit = logit.squeeze()
        char_values = logit.argmax(axis=1)
        transcribed_text = ""
        previous_char = ""
        for i in char_values:
            char = "abcdefghijklmnopqrstuvwxyz .'_-"[i]
            if char != previous_char:
                transcribed_text += char
            previous_char = char

        print(transcribed_text)


def createAcousticModel(session, hyper_params, batch_size, forward_only):
    num_labels = 31
    input_dim = 123
    model = AcousticModel(num_labels, hyper_params["num_layers"],
                          hyper_params["hidden_size"], hyper_params["dropout"],
                          batch_size, hyper_params["learning_rate"],
                          hyper_params["lr_decay_factor"], hyper_params["grad_clip"],
                          hyper_params["max_input_seq_length"],
                          hyper_params["max_target_seq_length"],
                          input_dim,
                          forward_only=forward_only)
    ckpt = tf.train.get_checkpoint_state(hyper_params["checkpoint_dir"])
    if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from {0}".format(ckpt.model_checkpoint_path))
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.initialize_all_variables())
    return model


def parse_args():
    """
    Parses the command line input.

    """
    DEFAULT_CONFIG_FILE = 'config.ini'

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_FILE,
                        help='Path to configuration file with hyper-parameters.')

    group = parser.add_mutually_exclusive_group(required=True)
    group.set_defaults(train=False)
    group.add_argument('--train', dest='train', action='store_true', help='Train the network')
    group.add_argument('--file', type=str, help='Path to a wav file to process')

    args = parser.parse_args()
    prog_params = {'file': args.file, 'config_file': args.config, 'train': args.train}
    return prog_params


if __name__ == "__main__":
    main()
