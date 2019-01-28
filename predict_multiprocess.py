import numpy as np
import os
from configparser import ConfigParser
from generator import AugmentedImageSequence
from models.keras import ModelFactory
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, precision_recall_curve, recall_score, \
    f1_score, precision_score, fbeta_score, accuracy_score
from utility import get_sample_counts
import pandas as pd
from utility import set_sess_cfg
from keras.utils import multi_gpu_model
import matplotlib.pyplot as plt
from math import ceil
from ReadDcmUtil import read_dcm
import shutil
import cv2
from multiprocessing import Pool


def main():
    for year in years:
        year = str(year)
        dcm_dir_year = os.path.join(dcm_dir, year)
        if not os.path.exists(dcm_dir_year):
            continue
        for month in months:
            month = str(month)
            dcm_dir_year_month = os.path.join(dcm_dir_year, month)
            if not os.path.exists(dcm_dir_year_month):
                continue
            for day in days:
                day = str(day)
                dcm_dir_year_month_day = os.path.join(dcm_dir_year_month, day)
                if not os.path.exists(dcm_dir_year_month_day):
                    continue
                for root, dirs, files in os.walk(dcm_dir_year_month_day):
                    pool = Pool(2)
                    pool.apply_async(predict_save,(root, dirs, files,))
                    pool.close()
                    pool.join()

    #save the predict result
    if len(files_names) >0 and len(predict_results)>0:
        results_path = os.path.join(pacs_results_dir, "results.csv")
        print(f"** write results to {results_path} **")
        results_df = pd.DataFrame(data={'files_names': files_names, 'predict_results': predict_results})
        results_df.to_csv(results_path, index=False)

def predict_save(root, dirs, files):
    for dcm in files:
        file_dcm = os.path.join(root, dcm)
        if not os.path.splitext(file_dcm)[1] == '.dcm':
            print(f'{file_dcm} is not a dicom file')
            continue
        img_array, img_file = read_dcm(file_dcm)
        y_hat = model_train.predict(img_array, verbose=1)
        print(f'prediction is : {y_hat}')
        file_path = os.path.splitext(file_dcm)[0]
        dir_names = file_path.split(os.path.sep)
        img_name = f'{year}_{month}_{day}_{dir_names[-2]}_{dir_names[-1]}.png'
        if y_hat < thresh_hold:
            save_path = os.path.join(normal_dir, img_name)
        else:
            save_path = os.path.join(abnormal_dir, img_name)
        cv2.imwrite(save_path, img_file, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
        files_names.append(file_dcm)
        predict_results.append(y_hat)
        print('**************************************************')

if __name__ == "__main__":
    # parser config
    config_file = "./config.ini"
    cp = ConfigParser()
    cp.read(config_file)
    image_dimension = cp["PREDICT"].getint("image_dimension")
    base_model_name = cp["PREDICT"].get("base_model_name")
    model_weights_path = cp["PREDICT"].get("model_weights_path")
    thresh_hold = cp["PREDICT"].getfloat("thresh_hold")
    grayscale = cp["PREDICT"].getboolean("grayscale")
    dcm_dir = cp["PREDICT"].get("dcm_dir")
    pacs_results_dir = cp["PREDICT"].get("pacs_results_dir")
    normal_dir = cp["PREDICT"].get("normal_dir")
    abnormal_dir = cp["PREDICT"].get("abnormal_dir")
    set_year = cp["PREDICT"].getint("set_year")
    set_month = cp["PREDICT"].getint("set_month")
    set_day = cp["PREDICT"].getint("set_day")
    if grayscale:
        channel = 1
    else:
        channel = 3
    # parse weights file path

    model_factory = ModelFactory()
    model = model_factory.get_model(
        model_name=base_model_name,
        use_base_weights=False,
        weights_path=model_weights_path,
        input_shape=(image_dimension, image_dimension, channel),
        transform_14=False)

    gpus = len(os.getenv("CUDA_VISIBLE_DEVICES", "0").split(","))
    if gpus > 1:
        print(f"** multi_gpu_model is used! gpus={gpus} **")
        model_train = multi_gpu_model(model, gpus)
    else:
        model_train = model

    print("** make prediction **")
    years = [2018, 2017, 2016]
    months = list(range(1, 13))
    days = list(range(17, 32))
    files_names = []
    predict_results = []
    np.set_printoptions(threshold=1e6)
    set_sess_cfg()
    main()
