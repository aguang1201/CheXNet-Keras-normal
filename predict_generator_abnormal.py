from __future__ import absolute_import, division, print_function

from os import environ
import os
import math
from configparser import ConfigParser
import keras
import numpy as np
import pandas as pd
import sklearn as skl
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.utils import multi_gpu_model
import matplotlib.pyplot as plt
from utility import set_sess_cfg
from models.keras import ModelFactory
import shutil


pd.set_option('display.max_rows', 20)
pd.set_option('precision', 4)
np.set_printoptions(precision=4)

environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Shut up tensorflow!
print("tf : {}".format(tf.__version__))
print("keras : {}".format(keras.__version__))
print("numpy : {}".format(np.__version__))
print("pandas : {}".format(pd.__version__))
print("sklearn : {}".format(skl.__version__))

# parser config
config_file = "./config.ini"
cp = ConfigParser()
cp.read(config_file)
image_dimension = cp["PREDICT"].getint("image_dimension")
base_model_name = cp["PREDICT"].get("base_model_name")
model_weights_path = cp["PREDICT"].get("model_weights_path")
thresh_hold = cp["PREDICT"].getfloat("thresh_hold")
grayscale = cp["PREDICT"].getboolean("grayscale")
pacs_dir = cp["PREDICT"].get("pacs_dir")
pacs_results_dir = cp["PREDICT"].get("pacs_results_dir")
batch_size = cp["PREDICT"].getint("batch_size")


def main():
    results_save_path = os.path.join(pacs_results_dir, 'results', 'nonchest_predicted_results.npy')
    if os.path.exists(results_save_path):
        y_probs = np.load(results_save_path)
    else:
        model_factory = ModelFactory()
        model = model_factory.get_model(
            model_name=base_model_name,
            use_base_weights=False,
            weights_path=model_weights_path,
            input_shape=(image_dimension, image_dimension, 3),
            transform_14=False)
        # model = load_model(model_nonchest_path)
        predict_datagen = ImageDataGenerator(rescale=1. / 255, featurewise_center=True, featurewise_std_normalization=True)
        predict_datagen.mean = np.array([0.485, 0.456, 0.406])
        predict_datagen.std = np.array([0.229, 0.224, 0.225])
        predict_generator = predict_datagen.flow_from_directory(
            pacs_dir,
            class_mode='binary',
            shuffle=False,
            target_size=(image_dimension, image_dimension),
            batch_size=batch_size)
        n_samples = predict_generator.samples
        filenames = predict_generator.filenames
        gpus = len(os.getenv("CUDA_VISIBLE_DEVICES", "0").split(","))
        if gpus > 1:
            print(f"** multi_gpu_model is used! gpus={gpus} **")
            model_train = multi_gpu_model(model, gpus)
        else:
            model_train = model
        y_probs = model_train.predict_generator(
            predict_generator,
            math.ceil(n_samples / batch_size),
            verbose=1,
            workers=1,
            use_multiprocessing=False)
        np.save(results_save_path, y_probs)

    for i, y_prob in enumerate(y_probs):
        dir_file = os.path.split(filenames[i])
        if y_prob < thresh_hold:
            save_path = os.path.join(pacs_dir, dir_file[0], "normal", dir_file[1])
        else:
            save_path = os.path.join(pacs_dir, dir_file[0], "abnormal", dir_file[1])
        shutil.move(os.path.join(pacs_dir, filenames[i]), save_path)

if __name__ == "__main__":
    np.set_printoptions(threshold=1e6)
    set_sess_cfg()
    main()
