import tensorflow as tf
import os
from model import train, test, averaged_test, test_entire_valid_set
from configuration import get_config

config = get_config()
tf.reset_default_graph()

if __name__ == "__main__":
    # start training
    if config.train:
        print("\nTraining Session")
        os.makedirs(config.model_path, exist_ok=True)
        train(config.model_path)
    # start test
    else:
        print("\nTest session")
        if os.path.isdir(config.model_path):
            # test(config.model_path)               # Original
            # averaged_test(config.model_path)      # Random average
            test_entire_valid_set(config.model_path)
        else:
            raise AssertionError("model path doesn't exist!")
