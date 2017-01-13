# coding=utf-8
"""
This is the main logic for serializing and deserializing dictionaries
of hyperparameters (for use in checkpoint restoration and sampling)
"""
import os
import pickle
import time
import logging
try:
    import ConfigParser as configparser
except ImportError:
    import configparser


class HyperParameterHandler(object):
    def __init__(self, config_file):
        """
        Retrieves hyper parameter information from either config file or checkpoint
        """
        self.hyper_params = self.read_config_file(config_file)

        # Set logging framework
        if self.hyper_params["log_file"] is not None:
            logging.basicConfig(filename=self.hyper_params["log_file"])
        logging.getLogger().setLevel(self.hyper_params["log_level"])

        logging.info("Using checkpoint %s", self.hyper_params["checkpoint_dir"])
        logging.debug("Using hyper params: %s", self.hyper_params)

        # Create checkpoint dir if needed
        if not os.path.exists(self.hyper_params["checkpoint_dir"]):
            os.makedirs(self.hyper_params["checkpoint_dir"])

        self.file_path = os.path.join(self.hyper_params["checkpoint_dir"], "hyperparams.p")
        if self.check_exists():
            if self.check_changed(self.hyper_params):
                if not self.hyper_params["use_config_file_if_checkpoint_exists"]:
                    self.hyper_params = self.get_params()
                    logging.info("Restoring hyper params from previous checkpoint...")
                else:
                    new_checkpoint_dir = "{0}_hidden_size_{1}_numlayers_{2}_signal_processing_{3}".format(
                        int(time.time()),
                        self.hyper_params["hidden_size"],
                        self.hyper_params["num_layers"],
                        self.hyper_params["signal_processing"])
                    new_checkpoint_dir = os.path.join(self.hyper_params["checkpoint_dir"],
                                                      new_checkpoint_dir)
                    os.makedirs(new_checkpoint_dir)
                    self.hyper_params["checkpoint_dir"] = new_checkpoint_dir
                    self.file_path = os.path.join(self.hyper_params["checkpoint_dir"], "hyperparams.p")
                    self.save_params(self.hyper_params)
            else:
                logging.info("No hyper parameter changed detected, using old checkpoint...")
        else:
            self.save_params(self.hyper_params)
            logging.info("No hyper params detected at checkpoint... reading config file")
        return

    def get_hyper_params(self):
        return self.hyper_params

    def save_params(self, dic):
        with open(self.file_path, 'wb') as handle:
            pickle.dump(dic, handle)

    def get_params(self):
        with open(self.file_path, 'rb') as handle:
            return pickle.load(handle)

    def check_exists(self):
        """
        Checks if hyper parameter file exists
        """
        return os.path.exists(self.file_path)

    def check_changed(self, new_params):
        if self.check_exists():
            old_params = self.get_params()
            # Default signal_processing to mfcc if not present for compatibility with old checkpoint files
            if "signal_processing" not in old_params:
                old_params["signal_processing"] = "mfcc"
            return old_params["num_layers"] != new_params["num_layers"] or\
                old_params["hidden_size"] != new_params["hidden_size"] or\
                old_params["signal_processing"] != new_params["signal_processing"]
        else:
            return False

    @staticmethod
    def read_config_file(config_file):
        """
        Reads in config file, returns dictionary of network params
        """
        config = configparser.ConfigParser()
        config.read(config_file)
        dic = {}
        acoustic_section = "acoustic_network_params"
        general_section = "general"
        training_section = "training"
        log_section = "logging"
        dic["num_layers"] = config.getint(acoustic_section, "num_layers")
        dic["hidden_size"] = config.getint(acoustic_section, "hidden_size")
        dic["dropout_input_keep_prob"] = config.getfloat(acoustic_section, "dropout_input_keep_prob")
        dic["dropout_output_keep_prob"] = config.getfloat(acoustic_section, "dropout_output_keep_prob")
        dic["batch_size"] = config.getint(acoustic_section, "batch_size")
        dic["learning_rate"] = config.getfloat(acoustic_section, "learning_rate")
        dic["lr_decay_factor"] = config.getfloat(acoustic_section, "lr_decay_factor")
        dic["grad_clip"] = config.getint(acoustic_section, "grad_clip")
        dic["signal_processing"] = config.get(acoustic_section, "signal_processing")
        dic["use_config_file_if_checkpoint_exists"] = config.getboolean(general_section,
                                                                        "use_config_file_if_checkpoint_exists")
        dic["steps_per_checkpoint"] = config.getint(general_section, "steps_per_checkpoint")
        dic["checkpoint_dir"] = config.get(general_section, "checkpoint_dir")
        dic["training_dataset_dirs"] = config.get(training_section, "training_dataset_dirs")
        dic["test_dataset_dirs"] = config.get(training_section, "test_dataset_dirs", fallback=None)
        dic["train_frac"] = config.getfloat(training_section, "train_frac", fallback=None)
        dic["max_input_seq_length"] = config.getint(training_section, "max_input_seq_length")
        dic["max_target_seq_length"] = config.getint(training_section, "max_target_seq_length")
        dic["tensorboard_dir"] = config.get(training_section, "tensorboard_dir", fallback=None)
        if dic["tensorboard_dir"] is not None and not os.path.exists(dic["tensorboard_dir"]):
            dic["tensorboard_dir"] = None
        dic["batch_normalization"] = config.getboolean(training_section, "batch_normalization", fallback=False)
        dic["dataset_size_ordering"] = config.getboolean(training_section, "dataset_size_ordering", fallback=False)
        dic["log_file"] = config.get(log_section, "log_file", fallback=None)
        log_level = config.get(log_section, "log_level", fallback='WARNING')
        dic["log_level"] = getattr(logging, log_level)
        if not isinstance(dic["log_level"], int):
            raise ValueError('Invalid log level: %s' % log_level)

        return dic
