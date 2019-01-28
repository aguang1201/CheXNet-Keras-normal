import numpy as np
import os
import pandas as pd
import tensorflow as tf
from keras import backend as K
import random


def set_sess_cfg():
    config_sess = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config_sess.gpu_options.allow_growth = True
    sess = tf.Session(config=config_sess)
    K.set_session(sess)


def get_sample_counts(output_dir, dataset):
    """
    Get total and class-wise positive sample count of a dataset

    Arguments:
    output_dir - str, folder of dataset.csv
    dataset - str, train|dev|test

    Returns:
    total_count - int
    class_positive_counts - dict of int, ex: {"Effusion": 300, "Infiltration": 500 ...}
    """
    df = pd.read_csv(os.path.join(output_dir, f"{dataset}.csv"))
    total_count = df.shape[0]
    # labels = df["Finding Labels"].as_matrix()
    labels = df["Finding Labels"].apply(lambda label: 0 if label == 'No Finding' else 1)
    positive_counts = np.sum(labels, axis=0)
    return total_count, positive_counts


def split_data_without_test(data_entry_file, class_names, train_patient_count, dev_patient_count,
               output_dir, random_state):
    """
    Create train and validate dataset split csv files

    """
    e = pd.read_csv(data_entry_file)

    # one hot encode
    for c in class_names:
        e[c] = e["Finding Labels"].apply(lambda labels: 1 if c in labels else 0)

    # shuffle and split
    pid = list(e["Patient ID"].unique())
    random.seed(random_state)
    random.shuffle(pid)
    train = e[e["Patient ID"].isin(pid[:train_patient_count])]
    dev = e[e["Patient ID"].isin(pid[train_patient_count:])]

    # export csv
    output_fields = ["Image Index", "Patient ID", "Finding Labels"] + class_names
    train[output_fields].to_csv(os.path.join(output_dir, "train.csv"), index=False)
    dev[output_fields].to_csv(os.path.join(output_dir, "dev.csv"), index=False)
    e[output_fields].to_csv(os.path.join(output_dir, "test.csv"), index=False)
    return