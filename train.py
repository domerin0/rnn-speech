'''
Main training routine for character level speech recognizer.

This entire project is based on the model described by the paper:
http://arxiv.org/pdf/1601.06581v2.pdf
'''

from tensorflow.python.platform import gfile
from models.AcousticModel import AcousticModel
import tensorflow as tf
import util.dataprocessor as dataprocessor
import util.hyperparams as hyperparams
try:
    import ConfigParser as configparser
except ImportError:
    import configparser
from math import floor

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("config_file", "config.ini", "Path to configuration file with hyper-parameters.")


def main():
    serializer = hyperparams.HyperParameterHandler(FLAGS.config_file)
    hyper_params = serializer.getHyperParams()

    data_processor = dataprocessor.DataProcessor(hyper_params["training_dataset_dir"])
    text_audio_pairs = data_processor.run()
    num_train = int(floor(hyper_params["train_frac"] * len(text_audio_pairs)))
    train_set = text_audio_pairs[:num_train]
    test_set = text_audio_pairs[num_train:]
    print("Using {0} size of test set".format(len(test_set)))

    with tf.Session() as sess:
        # create model
        print("Building model... (this takes a while)")
        model = createAcousticModel(sess, hyper_params)
        print("Setting up audio processor...")
        model.initializeAudioProcessor(hyper_params["max_input_seq_length"])
        print("Start training...")
        model.train(sess, test_set, train_set, hyper_params["steps_per_checkpoint"], hyper_params["checkpoint_dir"])


def createAcousticModel(session, hyper_params):
    num_labels = 31
    input_dim = 123
    model = AcousticModel(num_labels, hyper_params["num_layers"],
                          hyper_params["hidden_size"], hyper_params["dropout"],
                          hyper_params["batch_size"], hyper_params["learning_rate"],
                          hyper_params["lr_decay_factor"], hyper_params["grad_clip"],
                          hyper_params["max_input_seq_length"],
                          hyper_params["max_target_seq_length"],
                          input_dim,
                          forward_only=False)
    ckpt = tf.train.get_checkpoint_state(hyper_params["checkpoint_dir"])
    if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from {0}".format(ckpt.model_checkpoint_path))
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.initialize_all_variables())
    return model


if __name__ == "__main__":
    main()
