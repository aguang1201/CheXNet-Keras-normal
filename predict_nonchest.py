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
from ReadDcmUtil import read_dcm, read_dcm_fromC
import shutil
import cv2
from multiprocessing import Pool


def main():
    # parser config
    config_file = "./config.ini"
    cp = ConfigParser()
    cp.read(config_file)
    image_dimension = cp["PREDICT"].getint("image_dimension")
    base_model_name = cp["PREDICT"].get("base_model_name")
    weights_nonchest_path = cp["PREDICT"].get("weights_nonchest_path")
    thresh_hold_nonchest = cp["PREDICT"].getfloat("thresh_hold_nonchest")
    grayscale = cp["PREDICT"].getboolean("grayscale")
    dcm_dir = cp["PREDICT"].get("dcm_dir")
    is_fromC = cp["PREDICT"].getboolean("is_fromC")
    if is_fromC:
        chest_dir = cp["PREDICT"].get("pacs_chest_dir_fromC")
    else:
        chest_dir = cp["PREDICT"].get("pacs_chest_dir")

    nonchest_dir = cp["PREDICT"].get("pacs_nonchest_dir")
    if grayscale:
        channel = 1
    else:
        channel = 3
    # parse weights file path

    model_factory = ModelFactory()
    model = model_factory.get_model(
        model_name=base_model_name,
        use_base_weights=False,
        weights_path=weights_nonchest_path,
        input_shape=(image_dimension, image_dimension, channel),
        transform_14=False)

    gpus = len(os.getenv("CUDA_VISIBLE_DEVICES", "0").split(","))
    if gpus > 1:
        print(f"** multi_gpu_model is used! gpus={gpus} **")
        model_train = multi_gpu_model(model, gpus)
    else:
        model_train = model

    print("** make prediction **")
    years = [2016]
    # years = reversed(list(range(2014, 2019)))
    # months = reversed(list(range(1, 13)))
    # years = list(range(2015, 2018))
    months = reversed(list(range(10, 11)))
    days = reversed(list(range(15, 32)))
    # files_names = []
    # predict_results = []
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
                    for dcm in files:
                        if os.path.splitext(dcm)[1] != '.dcm':
                            print(f'{dcm} is not a dicom file')
                            continue
                        file_dcm = os.path.join(root, dcm)
                        if is_fromC:
                            img_array, img_file = read_dcm_fromC(file_dcm, out_put_size=1024)
                        else:
                            img_array, img_file = read_dcm(file_dcm, out_put_size=1024)
                        if img_array is not None:
                            y_hat = model_train.predict(img_array, verbose=1)
                            print(f'prediction is : {y_hat}')
                            file_path = os.path.splitext(file_dcm)[0]
                            dir_names = file_path.split(os.path.sep)
                            img_name = f'{year}_{month}_{day}_{dir_names[-2]}_{dir_names[-1]}.png'
                            if y_hat < thresh_hold_nonchest:
                                save_path = os.path.join(nonchest_dir, img_name)
                            else:
                                save_path = os.path.join(chest_dir, img_name)
                            cv2.imwrite(save_path, img_file, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
                            # files_names.append(file_dcm)
                            # predict_results.append(y_hat)
                            print(f'save image to {save_path}')
    #save the predict result
    # if len(files_names) >0 and len(predict_results)>0:
    #     results_path = os.path.join(pacs_results_dir, "results.csv")
    #     print(f"** write results to {results_path} **")
    #     results_df = pd.DataFrame(data={'files_names': files_names, 'predict_results': predict_results})
    #     results_df.to_csv(results_path, index=False)

if __name__ == "__main__":
    np.set_printoptions(threshold=1e6)
    set_sess_cfg()
    main()
