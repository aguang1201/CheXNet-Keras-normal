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
weights_nonchest_path = cp["PREDICT"].get("weights_nonchest_path")
thresh_hold_nonchest = cp["PREDICT"].getfloat("thresh_hold_nonchest")
grayscale = cp["PREDICT"].getboolean("grayscale")
pacs_dir = cp["PREDICT"].get("pacs_dir")
pacs_results_dir = cp["PREDICT"].get("pacs_results_dir")
batch_size = cp["PREDICT"].getint("batch_size")


def main():
    results_save_path = os.path.join(pacs_results_dir, 'results', 'nonchest_predicted_results.npy')
    predict_datagen = ImageDataGenerator(rescale=1. / 255, featurewise_center=True, featurewise_std_normalization=True)
    predict_datagen.mean = np.array([0.485, 0.456, 0.406])
    predict_datagen.std = np.array([0.229, 0.224, 0.225])
    predict_generator = predict_datagen.flow_from_directory(
        pacs_dir,
        class_mode='binary',
        shuffle=False,
        target_size=(image_dimension, image_dimension),
        batch_size=batch_size)
    filenames = predict_generator.filenames

    if os.path.exists(results_save_path):
        y_probs = np.load(results_save_path)
    else:
        model_factory = ModelFactory()
        model = model_factory.get_model(
            model_name=base_model_name,
            use_base_weights=False,
            weights_path=weights_nonchest_path,
            input_shape=(image_dimension, image_dimension, 3),
            transform_14=False)
        # model = load_model(model_nonchest_path)
        n_samples = predict_generator.samples
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

    print(f'y_probs shape:{y_probs.shape}')

    for i, y_prob in enumerate(y_probs):
        # print(f'y_prob:{y_prob},i:{i},filename:{filenames[i]}')
        dir_file = os.path.split(filenames[i])
        if y_prob < thresh_hold_nonchest:
            save_path = os.path.join(pacs_results_dir, dir_file[0], "nonchest", dir_file[1])
        else:
            save_path = os.path.join(pacs_results_dir, dir_file[0], "chest", dir_file[1])
        if os.path.exists(save_path):
            print(f"The image is exist:{save_path}")
            continue
        else:
            src_path = os.path.join(pacs_dir, filenames[i])
            print(f'y_prob:{y_prob},i:{i},src_path:{src_path},save_path:{save_path}')
            shutil.move(src_path, save_path)
            # shutil.copy(src_path, save_path)
        # if y_prob < thresh_hold_nonchest:
        #     save_path = '/media/room/COMMON/False_chest/nonchest'
        #     src_path = os.path.join(pacs_dir, filenames[i])
        #     print(f'y_prob:{y_prob},i:{i},src_path:{src_path},save_path:{save_path}')
            # shutil.move(src_path, save_path)

if __name__ == "__main__":
    np.set_printoptions(threshold=1e6)
    set_sess_cfg()
    main()
