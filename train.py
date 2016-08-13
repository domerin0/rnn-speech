'''
Main training routine for character level speech recognizer.

This entire project is based on the model described by the paper:
http://arxiv.org/pdf/1601.06581v2.pdf
'''

from tensorflow.python.platform import gfile
from models.AcousticModel import AcousticModel
import tensorflow as tf
import sys
import os
import time
import util.dataprocessor as dataprocessor
import util.hyperparams as hyperparams
try:
    import ConfigParser as configparser
except ImportError:
    import configparser
from multiprocessing import Process, Pipe
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
    # setting up piplines to be able to load data async (one for test set, one for train)
    # TODO tensorflow probably has something built in for this, look into it
    parent_train_conn, child_train_conn = Pipe()
    parent_test_conn, child_test_conn = Pipe()

    with tf.Session() as sess:
        # create model
        print("Building model... (this takes a while)")
        model = createAcousticModel(sess, hyper_params)
        print("Setting up audio processor...")
        model.initializeAudioProcessor(hyper_params["max_input_seq_length"])
        print("Setting up piplines to test and train data...")
        model.setConnections(child_test_conn, child_train_conn)

        num_test_batches = model.getNumBatches(test_set)
        num_train_batches = model.getNumBatches(train_set)

        train_batch_pointer = 0
        test_batch_pointer = 0

        async_train_loader = Process(
            target=model.getBatch,
            args=(train_set, train_batch_pointer, True))
        async_train_loader.start()

        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        while True:
            # begin timer
            start_time = time.time()
            # receive batch from pipe
            step_batch_inputs = parent_train_conn.recv()
            # async_train_loader.join()

            train_batch_pointer = step_batch_inputs[5] % num_train_batches

            # begin fetching other batch while graph processes previous one
            async_train_loader = Process(
                target=model.getBatch,
                args=(train_set, train_batch_pointer, True))
            async_train_loader.start()
            # print step_batch_inputs[0]
            _, step_loss = model.step(sess, step_batch_inputs[0], step_batch_inputs[1],
                                      step_batch_inputs[2], step_batch_inputs[3],
                                      step_batch_inputs[4], forward_only=False)
            # print _
            print("Step {0} with loss {1}".format(current_step, step_loss))
            step_time += (time.time() - start_time) / hyper_params["steps_per_checkpoint"]
            loss += step_loss / hyper_params["steps_per_checkpoint"]
            current_step += 1
            if current_step % hyper_params["steps_per_checkpoint"] == 0:
                print("global step %d learning rate %.4f step-time %.2f loss %.2f" %
                      (model.global_step.eval(), model.learning_rate.eval(), step_time, loss))
                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)

                checkpoint_path = os.path.join(hyper_params["checkpoint_dir"], "acousticmodel.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0
                # begin loading test data async
                # (uses different pipline than train data)
                async_test_loader = Process(
                    target=model.getBatch,
                    args=(test_set, test_batch_pointer, False))
                async_test_loader.start()
                print(num_test_batches)
                for i in range(num_test_batches):
                    print("On {0}th training iteration".format(i))
                    eval_inputs = parent_test_conn.recv()
                    # async_test_loader.join()
                    test_batch_pointer = eval_inputs[5] % num_test_batches
                    # tell audio processor to go get another batch ready
                    # while we run last one through the graph
                    if i != num_test_batches - 1:
                        async_test_loader = Process(
                            target=model.getBatch,
                            args=(test_set, test_batch_pointer, False))
                        async_test_loader.start()
                    _, loss = model.step(sess, eval_inputs[0], eval_inputs[1],
                                         eval_inputs[2], eval_inputs[3],
                                         eval_inputs[4], forward_only=True)
                print("\tTest: loss %.2f" % loss)
                sys.stdout.flush()


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
