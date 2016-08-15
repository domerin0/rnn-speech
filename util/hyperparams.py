'''
This is the main logic for serializing and deserializing dictionaries
of hyperparameters (for use in checkpoint restoration and sampling)
'''
import os
import pickle
import time
try:
    import ConfigParser as configparser
except ImportError:
    import configparser


class HyperParameterHandler(object):
    def __init__(self, config_file):
        '''
        Retrieves hyper parameter information from either config file or checkpoint
        '''
        self.hyper_params = self.readConfigFile(config_file)

        print("Using checkpoint {0}".format(self.hyper_params["checkpoint_dir"]))
        print("Using hyper params: {0}".format(self.hyper_params))

        # Create checkpoint dir if needed
        if not os.path.exists(self.hyper_params["checkpoint_dir"]):
            os.makedirs(self.hyper_params["checkpoint_dir"])

        self.file_path = os.path.join(self.hyper_params["checkpoint_dir"], "hyperparams.p")
        if self.checkExists():
            if self.checkChanged(self.hyper_params):
                if not self.hyper_params["use_config_file_if_checkpoint_exists"]:
                    self.hyper_params = self.getParams()
                    print("Restoring hyper params from previous checkpoint...")
                else:
                    new_checkpoint_dir = "{0}_hidden_size_{1}_numlayers_{2}_dropout_{3}".format(
                        int(time.time()),
                        self.hyper_params["hidden_size"],
                        self.hyper_params["num_layers"],
                        self.hyper_params["dropout"])
                    new_checkpoint_dir = os.path.join(self.hyper_params["checkpoint_dir"],
                                                      new_checkpoint_dir)
                    os.makedirs(new_checkpoint_dir)
                    self.hyper_params["checkpoint_dir"] = new_checkpoint_dir
                    self.file_path = os.path.join(self.hyper_params["checkpoint_dir"], "hyperparams.p")
                    self.saveParams(self.hyper_params)
            else:
                print("No hyper parameter changed detected, using old checkpoint...")
        else:
            self.saveParams(self.hyper_params)
            print("No hyper params detected at checkpoint... reading config file")
        return

    def getHyperParams(self):
        return self.hyper_params

    def saveParams(self, dic):
        with open(self.file_path, 'wb') as handle:
            pickle.dump(dic, handle)

    def getParams(self):
        with open(self.file_path, 'rb') as handle:
            return pickle.load(handle)

    def checkExists(self):
        '''
        Checks if hyper parameter file exists
        '''
        return os.path.exists(self.file_path)

    def checkChanged(self, new_params):
        if self.checkExists():
            old_params = self.getParams()
            return old_params["num_layers"] != new_params["num_layers"] or\
                old_params["hidden_size"] != new_params["hidden_size"] or\
                old_params["dropout"] != new_params["dropout"] or\
                old_params["max_input_seq_length"] != new_params["max_input_seq_length"] or\
                old_params["max_target_seq_length"] != new_params["max_target_seq_length"]or\
                old_params["batch_size"] != new_params["batch_size"]
        else:
            return False

    def readConfigFile(self, config_file):
        '''
        Reads in config file, returns dictionary of network params
        '''
        config = configparser.ConfigParser()
        config.read(config_file)
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
        dic["load_save_input_vec"] = config.getboolean(training_section, "load_save_input_vec", fallback=False)

        return dic