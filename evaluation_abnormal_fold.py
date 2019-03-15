import numpy as np
import os
from configparser import ConfigParser
from generator import AugmentedImageSequence
from models.keras import ModelFactory
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, precision_recall_curve, recall_score, \
    f1_score, precision_score, fbeta_score, accuracy_score, cohen_kappa_score
from utility import get_sample_counts
import pandas as pd
from utility import set_sess_cfg
from keras.utils import multi_gpu_model
import matplotlib.pyplot as plt
from math import ceil
from keras.preprocessing.image import ImageDataGenerator
import glob
from scipy import misc
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

pred_threshold_list = np.linspace(0, 1, 1001)


def crop_image(image, ratio=0.1):
    height, width = image.shape[:2]
    delta_x = int(width * ratio)
    delta_y = int(height * ratio)
    # image_crop = image[delta_y:height - delta_y, delta_x:width - delta_x, 0:3]
    image_crop = image[delta_y:height - 2*delta_y, delta_x:width - delta_x, 0:3]
    image_crop = misc.imresize(image_crop, image.shape)
    return image_crop.astype('float32')


def max_threshold(f1_score_list):
    return np.argmax(f1_score_list)

def recall_specificity_f1_score(recall, specificity):
    return 2 * (specificity * recall) / (specificity + recall)

def main():
    # parser config
    config_file = "./config.ini"
    cp = ConfigParser()
    cp.read(config_file)
    # default config
    image_generator_dir = cp["DEFAULT"].get("image_generator_dir")
    # test config
    image_dimension = cp["EVALUATION"].getint("image_dimension")
    base_model_name = cp["EVALUATION"].get("base_model_name")
    output_dir = cp["EVALUATION"].get("output_dir")
    weight_dir = cp["EVALUATION"].get("weight_dir")
    batch_size = cp["EVALUATION"].getint("batch_size")
    seed = cp["EVALUATION"].getint("seed")
    mean = cp["EVALUATION"].getfloat("mean")
    std = cp["EVALUATION"].getfloat("std")
    generator_workers = cp["EVALUATION"].getint("generator_workers")
    mode = cp["EVALUATION"].get("mode")
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    grayscale = cp["EVALUATION"].getboolean("grayscale")
    if grayscale:
        channel = 1
    else:
        channel = 3
    # parse weights file path
    evaluation_base_dir = os.path.join(output_dir, weight_dir)
    # best_weights_path = os.path.join(evaluation_base_dir, "best_weights.h5")
    best_weights_AUC_path = os.path.join(evaluation_base_dir, "best_weights_AUC.h5")
    # last_weights_path = os.path.join(evaluation_base_dir, "last_weights.h5")
    weights = [best_weights_AUC_path]
    # weights = glob.glob(f'{evaluation_base_dir}/*.h5')

    data_aug_dict = dict(
        rescale=1. / 255,
        featurewise_center=True,
        featurewise_std_normalization=True,
        # preprocessing_function=crop_image,
    )
    generator_train_data = ImageDataGenerator(**data_aug_dict)
    generator_train_data.mean = mean
    generator_train_data.std = std
    val_generator = generator_train_data.flow_from_directory(
        image_generator_dir,
        shuffle=False,  # otherwise we get distorted batch-wise metrics
        class_mode='binary',
        target_size=(image_dimension, image_dimension),
        batch_size=batch_size,
    )
    y = val_generator.classes
    for model_weights_path in weights:
        if os.path.exists(model_weights_path):
            evaluation_dir = os.path.join(evaluation_base_dir, 'evaluation', mode, os.path.basename(model_weights_path).split('.')[0])
            if not os.path.exists(evaluation_dir):
                os.makedirs(evaluation_dir)
            model_factory = ModelFactory()
            model = model_factory.get_model(
                model_name=base_model_name,
                use_base_weights=False,
                weights_path=model_weights_path,
                input_shape=(image_dimension, image_dimension, channel),
                transform_14=False,
                #add_full_connection=True,
            )
            print("** load test generator **")
            gpus = len(os.getenv("CUDA_VISIBLE_DEVICES", "").split(","))
            if gpus > 1:
                print(f"** multi_gpu_model is used! gpus={gpus} **")
                model_train = multi_gpu_model(model, gpus)
            else:
                model_train = model

            # if model_weights_path is not None:
            #     print(f"load model weights_path: {model_weights_path}")
            #     model_train.load_weights(model_weights_path)

            print("** make prediction **")
            # y_hat = model_train.predict_generator(val_generator, verbose=1, workers=1, use_multiprocessing=False)
            y_hat = model_train.predict_generator(val_generator, verbose=1, workers=generator_workers, use_multiprocessing=True)
            roc_auc_log_path = os.path.join(evaluation_dir, "roc_auc.log")
            print(f"** write log to {roc_auc_log_path} **")
            with open(roc_auc_log_path, "w") as f:
                try:
                    score = roc_auc_score(y, y_hat)
                except ValueError:
                    score = 0
                f.write(f"roc_auc_score: {score}\n")
                f.write("-------------------------\n")

            PR_dir = os.path.join(evaluation_dir, 'PR_Curve')
            ROC_dir = os.path.join(evaluation_dir, 'ROC_Curve')
            if not os.path.exists(PR_dir):
                os.makedirs(PR_dir)
            if not os.path.exists(ROC_dir):
                os.makedirs(ROC_dir)
            precision, recall, threshold = precision_recall_curve(y, y_hat)
            # 绘制P-R曲线
            plt.figure()
            plt.title('Precision-Recall Curve')
            plt.plot(recall, precision)
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlim([0.0, 1.10])
            plt.ylim([0.0, 1.10])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            # plt.show()
            plt.savefig(os.path.join(PR_dir, f"PR.png"))
            plt.close('all')

            # 计算fpr、tpr
            fpr, tpr, thresholds = roc_curve(y, y_hat)
            roc_auc = auc(fpr, tpr)
            # 绘制ROC曲线
            plt.figure()
            plt.title('Receiver Operating Characteristic')
            plt.plot(fpr, tpr, label='AUC = %.2f' % roc_auc)
            plt.legend(loc='lower right')
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlim([-0.1, 1.0])
            plt.ylim([-0.1, 1.01])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            # plt.show()
            plt.savefig(os.path.join(ROC_dir, f"ROC.png"))
            plt.close('all')

            lines = []
            for pred_threshold in pred_threshold_list:
                line_results = []
                y_pred = np.where(y_hat > pred_threshold, 1, 0)
                recall = recall_score(y, y_pred)
                precision = precision_score(y, y_pred)
                accuracy = accuracy_score(y, y_pred)
                confusion_matrix_score = confusion_matrix(y, y_pred)
                # TP = confusion_matrix_score[1, 1]
                TN = confusion_matrix_score[0, 0]
                FP = confusion_matrix_score[0, 1]
                # FN = confusion_matrix_score[1, 0]
                specificity = TN / float(TN + FP)
                cohen_kappa = cohen_kappa_score(y, y_pred)
                recall_specificity_f1 = recall_specificity_f1_score(recall, specificity)
                f1 = f1_score(y, y_pred)
                line_results.extend([confusion_matrix_score, accuracy, precision, recall, specificity, recall_specificity_f1, f1, cohen_kappa])
                lines.append(line_results)
            columns_list = []
            columns_list.extend(['confusion_matrix', 'accuracy', 'precision', 'recall', 'specificity', 'recall_specificity_f1', 'f1', 'cohen_kappa'])
            precision_recall_f1_log_path = os.path.join(evaluation_dir, "precision_recall_f1.csv")
            precision_recall_f1 = pd.DataFrame(data=lines, index=pred_threshold_list, columns=columns_list)
            precision_recall_f1.index.names = ['threshhold']
            precision_recall_f1.to_csv(precision_recall_f1_log_path)
            columns_sortby = ['precision', 'recall', 'accuracy', 'specificity', 'recall_specificity_f1', 'f1']
            for column_sortby in columns_sortby:
                precision_recall_f1.sort_values(by=[column_sortby, 'threshhold'], inplace=True,
                                                ascending=[False, True])
                df_sorted_path = os.path.join(evaluation_dir, f"sorted_by_{column_sortby}.csv")
                precision_recall_f1.to_csv(df_sorted_path)


if __name__ == "__main__":
    np.set_printoptions(threshold=1e6)
    set_sess_cfg()
    main()
