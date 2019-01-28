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


pred_threshold_list = np.linspace(0, 1, 1001)



def max_threshold(f1_score_list):
    return np.argmax(f1_score_list)


def main():
    # parser config
    config_file = "./config.ini"
    cp = ConfigParser()
    cp.read(config_file)
    # default config
    image_source_dir = cp["DEFAULT"].get("image_source_dir")
    # test config
    image_dimension = cp["EVALUATION"].getint("image_dimension")
    base_model_name = cp["EVALUATION"].get("base_model_name")
    output_dir = cp["EVALUATION"].get("output_dir")
    weight_dir = cp["EVALUATION"].get("weight_dir")
    batch_size = cp["EVALUATION"].getint("batch_size")
    test_steps = cp["EVALUATION"].get("test_steps")
    use_best_weights = cp["EVALUATION"].getboolean("use_best_weights")
    grayscale = cp["EVALUATION"].getboolean("grayscale")
    if grayscale:
        channel = 1
    else:
        channel = 3
    # parse weights file path
    evaluation_dir = os.path.join(output_dir, weight_dir)
    output_weights_name = cp["TRAIN"].get("output_weights_name")
    weights_path = os.path.join(evaluation_dir, output_weights_name)
    best_weights_path = os.path.join(evaluation_dir, f"best_{output_weights_name}")

    # get test sample count
    test_counts, _ = get_sample_counts(output_dir, "test")

    # compute steps
    if test_steps == "auto":
        test_steps = ceil(test_counts / batch_size)
    else:
        try:
            test_steps = int(test_steps)
        except ValueError:
            raise ValueError(f"""
                test_steps: {test_steps} is invalid,
                please use 'auto' or integer.
                """)
    print(f"** test_steps: {test_steps} **")

    print("** load model **")
    if use_best_weights:
        print("** use best weights **")
        model_weights_path = best_weights_path
    else:
        print("** use last weights **")
        model_weights_path = weights_path
    model_factory = ModelFactory()
    model = model_factory.get_model(
        model_name=base_model_name,
        use_base_weights=False,
        weights_path=model_weights_path,
        input_shape=(image_dimension, image_dimension, channel),
        transform_14=False)

    print("** load test generator **")
    test_sequence = AugmentedImageSequence(
        # dataset_csv_file=os.path.join(output_dir, "dev.csv"),
        dataset_csv_file=os.path.join(output_dir, "test.csv"),
        source_image_dir=image_source_dir,
        batch_size=batch_size,
        target_size=(image_dimension, image_dimension),
        augmenter=None,
        steps=test_steps,
        shuffle_on_epoch_end=False,
        grayscale=grayscale,
    )

    gpus = len(os.getenv("CUDA_VISIBLE_DEVICES", "0").split(","))
    if gpus > 1:
        print(f"** multi_gpu_model is used! gpus={gpus} **")
        model_train = multi_gpu_model(model, gpus)
    else:
        model_train = model

    print("** make prediction **")
    y_hat = model_train.predict_generator(test_sequence, verbose=1, workers=8)
    y = test_sequence.get_y_true()

    #save the predict result
    results_path = os.path.join(evaluation_dir, "results.csv")
    print(f"** write results to {results_path} **")
    test_file = os.path.join(output_dir, "test.csv")
    test_df = pd.read_csv(test_file)
    results_df = pd.DataFrame(data=y_hat, columns=['abnormal'])
    results_df['id'] = test_df['Image Index']
    cols = results_df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    results_df[cols].to_csv(results_path, index=False)

    roc_auc_log_path = os.path.join(evaluation_dir, "roc_auc.log")
    print(f"** write log to {roc_auc_log_path} **")
    with open(roc_auc_log_path, "w") as f:
        try:
            score = roc_auc_score(y, y_hat)
        except ValueError:
            score = 0
        f.write(f"roc_auc_score: {score}\n")
        f.write("-------------------------\n")

    precision_recall_threshold_path = os.path.join(evaluation_dir, "precision_recall_threshold.log")
    print(f"** write precision_recall_threshold to {precision_recall_threshold_path} **")
    PR_dir = os.path.join(evaluation_dir, 'PR_Curve')
    ROC_dir = os.path.join(evaluation_dir, 'ROC_Curve')
    if not os.path.exists(PR_dir):
        os.makedirs(PR_dir)
    if not os.path.exists(ROC_dir):
        os.makedirs(ROC_dir)

    try:
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
    except ValueError:
        precision = 0
        recall = 0
        threshold = 0
    with open(precision_recall_threshold_path, "w") as f:
        f.write(f"precision: {precision}\n")
        f.write(f"recall: {recall}\n")
        f.write(f"threshold: {threshold}\n")
    lines = []
    for pred_threshold in pred_threshold_list:
        line_results = []
        y_pred = np.where(y_hat > pred_threshold, 1, 0)
        fbeta_list = []
        fbeta_arange = list(np.arange(1, 2.1, 0.1))
        for beta in fbeta_arange:
            fbeta = fbeta_score(y, y_pred, beta=beta)
            fbeta_list.append(fbeta)
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
        line_results.extend([precision, recall, accuracy, confusion_matrix_score, specificity, cohen_kappa])
        line_results.extend(fbeta_list)
        lines.append(line_results)
    fbeta_col_list = ['beta' + str(float('%.1f' % beta)) for beta in fbeta_arange]
    columns_list = []
    columns_list.extend(['precision', 'recall', 'accuracy', 'confusion_matrix',  'specificity', 'cohen_kappa'])
    columns_list.extend(fbeta_col_list)
    precision_recall_f1_log_path = os.path.join(evaluation_dir, "precision_recall_f1.csv")
    precision_recall_f1 = pd.DataFrame(data=lines, index=pred_threshold_list, columns=columns_list)
    precision_recall_f1.to_csv(precision_recall_f1_log_path)

    columns_sortby = ['precision', 'recall', 'accuracy']
    columns_sortby.extend(fbeta_col_list)
    for column_sortby in columns_sortby:
        precision_recall_f1.sort_values(by=column_sortby, inplace=True, ascending=False)
        df_sorted_path = os.path.join(evaluation_dir, f"sorted_by_{column_sortby}.csv")
        precision_recall_f1.to_csv(df_sorted_path)
    columns_list_thresh = ['threshold_max']
    columns_list_thresh.extend(columns_list)
    fbeta_lines = []
    fbeta_columns = []
    for fbeta_col in fbeta_col_list:
        max_fbeta_index = np.argmax(precision_recall_f1[fbeta_col])
        fbeta_max_list = []
        fbeta_max_list.extend([fbeta_col, max_fbeta_index])
        fbeta_max_list.extend(precision_recall_f1.loc[max_fbeta_index])
        fbeta_lines.append(fbeta_max_list)
    fbeta_columns.append('fbeta')
    fbeta_columns.extend(columns_list_thresh)
    fbeta_max_path = os.path.join(evaluation_dir, "max_fbeta.csv")
    df_fbeta_max = pd.DataFrame(data=fbeta_lines, columns=fbeta_columns)
    df_fbeta_max.to_csv(fbeta_max_path, index=False)

if __name__ == "__main__":
    np.set_printoptions(threshold=1e6)
    set_sess_cfg()
    main()
