'''
Main program to use the speech recognizer.
'''

from tensorflow.python.platform import gfile
from models.AcousticModel import AcousticModel
import tensorflow as tf
import util.hyperparams as hyperparams
import util.audioprocessor as audioprocessor
import argparse


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("config_file", "config.ini", "Path to configuration file with hyper-parameters.")


def main():
    serializer = hyperparams.HyperParameterHandler(FLAGS.config_file)
    hyper_params = serializer.getHyperParams()
    max_input_seq_length = hyper_params["max_input_seq_length"]
    audio_processor = audioprocessor.AudioProcessor(max_input_seq_length)
    prog_params = parse_args()

    feat_vec, original_feat_vec_length = audio_processor.processFLACAudio(prog_params['file'])
    if original_feat_vec_length > max_input_seq_length:
        print("File too long")
        return

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
