'''
Main program to use the speech recognizer.
'''

from tensorflow.python.platform import gfile
from models.AcousticModel import AcousticModel
import tensorflow as tf
import os
import time
import util.hyperparams as hyperparams
import util.audioprocessor as audioprocessor
try:
    import ConfigParser as configparser
except ImportError:
    import configparser
import argparse


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("config_file", "config.ini", "Path to configuration file with hyper-parameters.")


def main():
    hyper_params = checkGetHyperParamDic()
    max_input_seq_length = hyper_params["max_input_seq_length"]
    audio_processor = audioprocessor.AudioProcessor(max_input_seq_length)
    prog_params = parse_args()

    feat_vec, original_feat_vec_length = audio_processor.processFLACAudio(prog_params['file'])
    if original_feat_vec_length > max_input_seq_length:
        print("File too long")
        return

    print("Using checkpoint {0}".format(hyper_params["checkpoint_dir"]))
    print("Using hyper params: {0}".format(hyper_params))

    with tf.Session() as sess:
        # create model
        print("Building model... (this takes a while)")
        model = createAcousticModel(sess, hyper_params)

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


def createAcousticModel(session, hyper_params):
    num_labels = 31
    input_dim = 123
    model = AcousticModel(num_labels, hyper_params["num_layers"],
                          hyper_params["hidden_size"], hyper_params["dropout"],
                          1, hyper_params["learning_rate"],
                          hyper_params["lr_decay_factor"], hyper_params["grad_clip"],
                          hyper_params["max_input_seq_length"],
                          hyper_params["max_target_seq_length"],
                          input_dim,
                          forward_only=True)
    ckpt = tf.train.get_checkpoint_state(hyper_params["checkpoint_dir"])
    if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from {0}".format(ckpt.model_checkpoint_path))
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.initialize_all_variables())
    return model


def checkGetHyperParamDic():
    '''
    Retrieves hyper parameter information from either config file or checkpoint
    '''
    hyper_params = readConfigFile()
    if not os.path.exists(hyper_params["checkpoint_dir"]):
        os.makedirs(hyper_params["checkpoint_dir"])
    serializer = hyperparams.HyperParameterHandler(hyper_params["checkpoint_dir"])
    if serializer.checkExists():
        if serializer.checkChanged(hyper_params):
            if not hyper_params["use_config_file_if_checkpoint_exists"]:
                hyper_params = serializer.getParams()
                print("Restoring hyper params from previous checkpoint...")
            else:
                new_checkpoint_dir = "{0}_hidden_size_{1}_numlayers_{2}_dropout_{3}".format(
                    int(time.time()),
                    hyper_params["hidden_size"],
                    hyper_params["num_layers"],
                    hyper_params["dropout"])
                new_checkpoint_dir = os.path.join(hyper_params["checkpoint_dir"],
                                                  new_checkpoint_dir)
                os.makedirs(new_checkpoint_dir)
                hyper_params["checkpoint_dir"] = new_checkpoint_dir
                serializer = hyperparams.HyperParameterHandler(hyper_params["checkpoint_dir"])
                serializer.saveParams(hyper_params)
        else:
            print("No hyper parameter changed detected, using old checkpoint...")
    else:
        serializer.saveParams(hyper_params)
        print("No hyper params detected at checkpoint... reading config file")
    return hyper_params


def readConfigFile():
    '''
    Reads in config file, returns dictionary of network params
    '''
    config = configparser.ConfigParser()
    config.read(FLAGS.config_file)
    dic = {}
    acoustic_section = "acoustic_network_params"
    general_section = "general"
    training_section = "training"
    dic["num_layers"] = config.getint(acoustic_section, "num_layers")
    dic["hidden_size"] = config.getint(acoustic_section, "hidden_size")
    dic["dropout"] = config.getfloat(acoustic_section, "dropout")
    dic["batch_size"] = config.getint(acoustic_section, "batch_size")
    dic["train_frac"] = config.getfloat(acoustic_section, "train_frac")
    dic["learning_rate"] = config.getfloat(acoustic_section, "learning_rate")
    dic["lr_decay_factor"] = config.getfloat(acoustic_section, "lr_decay_factor")
    dic["grad_clip"] = config.getint(acoustic_section, "grad_clip")
    dic["use_config_file_if_checkpoint_exists"] = config.getboolean(general_section,
                                                                    "use_config_file_if_checkpoint_exists")
    dic["steps_per_checkpoint"] = config.getint(general_section, "steps_per_checkpoint")
    dic["checkpoint_dir"] = config.get(general_section, "checkpoint_dir")
    dic["training_dataset_dir"] = config.get(training_section, "training_dataset_dir")
    dic["max_input_seq_length"] = config.getint(training_section, "max_input_seq_length")
    dic["max_target_seq_length"] = config.getint(training_section, "max_target_seq_length")
    return dic


def parse_args():
    """
    Parses the command line input.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='wav file to process')
    args = parser.parse_args()
    prog_params = {'file': args.file}
    return prog_params


if __name__ == "__main__":
    main()
