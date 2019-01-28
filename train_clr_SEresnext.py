import json
import shutil
import os
import pickle
from callback import MultiGPUModelCheckpoint, Callback_AUROC_ImageDataGenerator
from configparser import ConfigParser
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping, LearningRateScheduler,CSVLogger
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from keras.metrics import binary_accuracy, binary_crossentropy
from metrics import mcor, precision, recall, f1
from models.keras import ModelFactory
from utility import set_sess_cfg
from sklearn.utils import class_weight
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from math import ceil
from PIL import ImageFile
from losses import focal_loss
from models.se_resnext import get_model
from clr_callback import *
import matplotlib.pyplot as plt


ImageFile.LOAD_TRUNCATED_IMAGES = True


# parser config
config_file = "./config.ini"
cp = ConfigParser()
cp.read(config_file)
# default config
output_dir = cp["DEFAULT"].get("output_dir")
image_generator_dir = cp["DEFAULT"].get("image_generator_dir")
output_weights_name = cp["TRAIN"].get("output_weights_name")
# train config
use_trained_model_weights = cp["TRAIN"].getboolean("use_trained_model_weights")
epochs = cp["TRAIN"].getint("epochs")
batch_size = cp["TRAIN"].getint("batch_size")
initial_learning_rate = cp["TRAIN"].getfloat("initial_learning_rate")
generator_workers = cp["TRAIN"].getint("generator_workers")
image_dimension = cp["TRAIN"].getint("image_dimension")
patience_reduce_lr = cp["TRAIN"].getint("patience_reduce_lr")
factor_reduce_lr = cp["TRAIN"].getfloat("factor_reduce_lr")
min_lr = cp["TRAIN"].getfloat("min_lr")
patience_early_stop = cp["TRAIN"].getint("patience_early_stop")
show_model_summary = cp["TRAIN"].getboolean("show_model_summary")
decay = cp["TRAIN"].getfloat("decay")
seed = cp["TRAIN"].getint("seed")
validation_split = cp["TRAIN"].getfloat("validation_split")
decay = cp["TRAIN"].getfloat("decay")
finetuning = cp["TRAIN"].getboolean("finetuning")
freezen_layers_block = cp["TRAIN"].getint("freezen_layers_block")
mean = cp["TRAIN"].getfloat("mean")
std = cp["TRAIN"].getfloat("std")
base_lr = cp["TRAIN"].getfloat("base_lr")
max_lr = cp["TRAIN"].getfloat("max_lr")


def main():
    # check output_dir, create it if not exists
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    running_flag_file = os.path.join(output_dir, ".training.lock")
    if os.path.isfile(running_flag_file):
        raise RuntimeError("A process is running in this directory!!!")
    else:
        open(running_flag_file, "a").close()
    try:
        print(f"backup config file to {output_dir}")
        shutil.copy(config_file, os.path.join(output_dir, os.path.split(config_file)[1]))
        python_file = os.path.basename(__file__)
        shutil.copy(python_file, os.path.join(output_dir, python_file))
        data_aug_dict = dict(
            rescale=1. / 255,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            featurewise_center=True,
            featurewise_std_normalization=True,
            fill_mode='constant',
        )
        generator_train_data = ImageDataGenerator(**data_aug_dict, validation_split=validation_split)
        generator_train_data.mean = mean
        generator_train_data.std = std
        train_generator = generator_train_data.flow_from_directory(
            image_generator_dir,
            shuffle=True,
            target_size=(image_dimension, image_dimension),
            class_mode='binary',
            batch_size=batch_size,
            subset='training',
            seed=seed)

        val_generator = generator_train_data.flow_from_directory(
            image_generator_dir,
            shuffle=False,  # otherwise we get distorted batch-wise metrics
            class_mode='binary',
            target_size=(image_dimension, image_dimension),
            batch_size=batch_size,
            subset='validation',
            seed=seed)

        classes = len(train_generator.class_indices)
        assert classes > 0
        n_of_train_samples = train_generator.samples
        n_of_val_samples = val_generator.samples

        # compute class weights
        print("** compute class weights from training data **")
        weights = class_weight.compute_class_weight('balanced', np.unique(train_generator.classes), train_generator.classes)
        weights = {0: weights[0], 1: weights[1]}
        print("** class_weights **")
        print(weights)

        print("** load model **")
        if use_trained_model_weights:
            # load training status for resuming
            training_stats_file = os.path.join(output_dir, ".training_stats.json")
            if os.path.isfile(training_stats_file):
                # TODO: add loading previous learning rate?
                training_stats = json.load(open(training_stats_file))
            else:
                training_stats = {}
            model_weights_file = os.path.join(output_dir, "best_weights_AUC.h5")
        else:
            training_stats = {}
            model_weights_file = None

        model = get_model(model_weights_file, target_size=image_dimension)

        if finetuning and freezen_layers_block:
            for layer in model.layers[:(427 - freezen_layers_block * 7)]:
                layer.trainable = False

        if show_model_summary:
            print(model.summary())

        output_best_weights = os.path.join(output_dir, f"best_{output_weights_name}")
        auc_best_weights = os.path.join(output_dir, f"best_auc_{output_weights_name}")
        # print(f"** set output weights path to: {output_weights_path} **")

        print("** check multiple gpu availability **")
        gpus = len(os.getenv("CUDA_VISIBLE_DEVICES", "0").split(","))
        if gpus > 1:
            print(f"** multi_gpu_model is used! gpus={gpus} **")
            model_train = multi_gpu_model(model, gpus)
            # FIXME: currently (Keras 2.1.2) checkpoint doesn't work with multi_gpu_model
            checkpoint = MultiGPUModelCheckpoint(
                filepath=output_best_weights,
                base_model=model,
                # save_weights_only=True,
                # save_best_only=True,
                verbose=1,
                # monitor='val_fmeasure',
                # monitor='val_acc',
                # monitor='f1',
            )
        else:
            model_train = model
            checkpoint = ModelCheckpoint(
                output_best_weights,
                 save_weights_only=True,
                 save_best_only=True,
                 verbose=1,
                 # monitor='val_fmeasure',
                # monitor='val_acc',
                # monitor='f1',
            )

        print("** compile model with class weights **")
        optimizer = Adam()
        # model_train.compile(optimizer=optimizer, loss=binary_crossentropy, metrics=[binary_accuracy, mcor, precision, recall, f1])
        model_train.compile(optimizer=optimizer, loss=focal_loss(gamma=2., alpha=.75), metrics=[binary_accuracy, mcor, precision, recall, f1])
        auroc = Callback_AUROC_ImageDataGenerator(
            sequence=val_generator,
            weights_path=auc_best_weights,
            stats=training_stats,
            workers=generator_workers,
        )
        csv_logger = CSVLogger(os.path.join(output_dir, 'training.csv'))
        batch_size_cycliclr = ceil(n_of_train_samples // batch_size // 2)
        print(f'batch_size_cycliclr is : {batch_size_cycliclr}')
        clr = CyclicLR(mode='exp_range', step_size=batch_size_cycliclr, base_lr=base_lr, max_lr=max_lr, gamma=0.99994)
        callbacks = [
            checkpoint,
            clr,
            TensorBoard(log_dir=os.path.join(output_dir, "logs"), batch_size=batch_size),
            csv_logger,
            auroc,
        ]

        print("** start training **")
        history = model_train.fit_generator(
            generator=train_generator,
            steps_per_epoch=ceil(n_of_train_samples // batch_size),
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=ceil(n_of_val_samples // batch_size),
            class_weight=weights,
            workers=generator_workers,
            use_multiprocessing=True,
            callbacks=callbacks)
        model_train.save(os.path.join(output_dir, "last_model.h5"))
        model_train.save_weights(os.path.join(output_dir, "last_weights.h5"))
        # dump history
        print("** dump history **")
        with open(os.path.join(output_dir, "history.pkl"), "wb") as f:
            pickle.dump({"history": history.history}, f)
        print("** done! **")

        h = clr.history
        lr = h['lr']
        iterations = h['iterations']
        plt.xlabel('Training Iterations')
        plt.ylabel('Learning Rate')
        plt.title("CLR - 'triangular' Policy")
        plt.plot(iterations, lr)
        plt.savefig(os.path.join(output_dir, 'iterations_lr.png'))
    finally:
        os.remove(running_flag_file)

if __name__ == "__main__":
    set_sess_cfg()
    main()
