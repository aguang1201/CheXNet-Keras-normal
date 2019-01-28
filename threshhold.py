import numpy as np
import os
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, precision_recall_curve, recall_score, \
    f1_score, precision_score, fbeta_score
import pandas as pd
from utility import set_sess_cfg
import matplotlib.pyplot as plt
from datetime import datetime

pred_threshold_list = np.linspace(0, 1, 1001)
beta = 2

csv_dir = 'report_csv'
csv_predict = 'front_pdata_report.csv'
csv_label = 'front_report.csv'


def max_threshold(f1_score_list):
    max_index = np.argmax(f1_score_list)
    return max_index, max(f1_score_list)


def main():
    output_dir = os.path.join('threshhold', f'{datetime.now()}')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    y_df = pd.read_csv(os.path.join(csv_dir, csv_label))
    y_hat_df = pd.read_csv(os.path.join(csv_dir, csv_predict))
    roc_auc_log_path = os.path.join(output_dir, "roc_auc.log")
    number_each_disease_file = os.path.join(output_dir, "number_each_disease.log")
    class_names = ['Abnormal', 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                   'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
    y = y_df[class_names].values
    y_hat = y_hat_df[class_names].fillna(value=0).values

    print(f"** write sum number of deseases to {number_each_disease_file} **")
    number_each_disease_arr = np.sum(y, axis=0)
    with open(number_each_disease_file, "w") as f:
        for class_name, number_each_disease in zip(class_names, number_each_disease_arr):
            f.write(f"The number of {class_name} is: {number_each_disease}\n")

    print(f"** write log to {roc_auc_log_path} **")
    aurocs = []
    with open(roc_auc_log_path, "w") as f:
        for i in range(len(class_names)):
            try:
                score = roc_auc_score(y[:, i], y_hat[:, i])
                aurocs.append(score)
            except ValueError:
                score = 0
            f.write(f"{class_names[i]}: {score}\n")
        mean_auroc = np.mean(aurocs)
        f.write("-------------------------\n")
        f.write(f"mean auroc: {mean_auroc}\n")
        print(f"mean auroc: {mean_auroc}")

    PR_dir = os.path.join(output_dir, 'PR_Curve')
    ROC_dir = os.path.join(output_dir, 'ROC_Curve')
    if not os.path.exists(PR_dir):
        os.makedirs(PR_dir)
    if not os.path.exists(ROC_dir):
        os.makedirs(ROC_dir)
    for i in range(len(class_names)):
        precision, recall, threshold = precision_recall_curve(y[:, i], y_hat[:, i])
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
        plt.savefig(os.path.join(PR_dir, f"PR_{class_names[i]}.png"))
        plt.close('all')

        # 计算fpr、tpr
        fpr, tpr, thresholds = roc_curve(y[:, i], y_hat[:, i])
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
        plt.savefig(os.path.join(ROC_dir, f"ROC_{class_names[i]}.png"))
        plt.close('all')

    f1_score_log_path = os.path.join(output_dir, "f1_score.csv")
    fbeta_score_log_path = os.path.join(output_dir, f"fbeta_{beta}_score.csv")
    recall_log_path = os.path.join(output_dir, "recall.csv")
    precision_log_path = os.path.join(output_dir, "precision.csv")
    confusion_matrix_log_path = os.path.join(output_dir, "confusion_matrix.csv")
    f1_score_all = []
    fbeta_score_all = []
    recall_all = []
    precision_all = []
    confusion_matrix_all = []
    # pred_threshold_list = [i / 10.0 for i in range(3, 10, 1)]

    for pred_threshold in pred_threshold_list:
        f1_score_line = []
        fbeta_score_line = []
        recall_line = []
        precision_line = []
        confusion_matrix_line = []
        for i in range(len(class_names)):
            y_pred = np.where(y_hat[:, i] > pred_threshold, 1, 0)
            f1 = f1_score(y[:, i], y_pred)
            f1_score_line.append(f1)
            fbeta = fbeta_score(y[:, i], y_pred, beta=beta)
            fbeta_score_line.append(fbeta)
            recall = recall_score(y[:, i], y_pred)
            recall_line.append(recall)
            precision = precision_score(y[:, i], y_pred)
            precision_line.append(precision)
            confusion_matrix_score = confusion_matrix(y[:, i], y_pred)
            confusion_matrix_line.append(confusion_matrix_score)
        f1_score_all.append(f1_score_line)
        fbeta_score_all.append(fbeta_score_line)
        recall_all.append(recall_line)
        precision_all.append(precision_line)
        confusion_matrix_all.append(confusion_matrix_line)
    df_f1_score = pd.DataFrame(data=f1_score_all, index=pred_threshold_list, columns=class_names)
    df_f1_score.to_csv(f1_score_log_path)
    df_fbeta_score = pd.DataFrame(data=fbeta_score_all, index=pred_threshold_list, columns=class_names)
    df_fbeta_score.to_csv(fbeta_score_log_path)
    df_recall = pd.DataFrame(data=recall_all, index=pred_threshold_list, columns=class_names)
    df_recall.to_csv(recall_log_path)
    df_precision = pd.DataFrame(data=precision_all, index=pred_threshold_list, columns=class_names)
    df_precision.to_csv(precision_log_path)
    df_confusion_matrix = pd.DataFrame(data=confusion_matrix_all, index=pred_threshold_list, columns=class_names)
    df_confusion_matrix.to_csv(confusion_matrix_log_path)

    f1_score_max_list = df_f1_score.apply(max_threshold)
    f1_score_max_path = os.path.join(output_dir, "f1_score_max.csv")
    df_f1_score_max = pd.DataFrame(data=list(f1_score_max_list), columns=['threshold_max', 'f1_score_max'])
    df_f1_score_max['class_names'] = class_names
    df_f1_score_max[['class_names', 'f1_score_max', 'threshold_max']].to_csv(f1_score_max_path, index=False)


if __name__ == "__main__":
    np.set_printoptions(threshold=1e6)
    main()
