import pandas as pd
import numpy as np
import os
from configparser import ConfigParser
import shutil


config_file = "./config.ini"
cp = ConfigParser()
cp.read(config_file)
pacs_results_dir = cp["PREDICT"].get("pacs_results_dir")
save_dir = os.path.join(pacs_results_dir, 'compared_with_hospital_results')

def read_xlsx_file():
    xls_dir = os.path.join(pacs_results_dir, 'hospital_results')
    xls_files_list = list(range(2014, 2019))
    result_df_list = []
    for xls_file in xls_files_list:
        pickout_file = os.path.join(xls_dir, str(xls_file) + '_pickout.csv')
        if os.path.exists(pickout_file):
            result_df = pd.read_csv(pickout_file)
        else:
            df_xls = pd.read_excel(os.path.join(xls_dir, str(xls_file) + '.xlsx'))
            df_xls['POSITIVE_FLAG_XLSX'] = df_xls['POSITIVE_FLAG'].apply(lambda flag: 0 if flag == '阴性' else 1)
            result_columns = ['RECORD_NO', 'POSITIVE_FLAG_XLSX']
            result_df = df_xls.loc[:, result_columns]
            result_df.to_csv(pickout_file, index=False)
        result_df_list.append(result_df)
    results_df_concated = pd.concat(result_df_list, ignore_index=True)
    return results_df_concated

def read_predicted_file():
    sub_dirs = ['normal', 'abnormal']
    result_df_list = []
    for i, sub_dir in enumerate(sub_dirs):
        result_file = os.path.join(pacs_results_dir, 'results', f'{sub_dir}.csv')
        if os.path.exists(result_file):
            df_results = pd.read_csv(result_file)
        else:
            pic_dir = os.path.join(pacs_results_dir, sub_dir, 'chest')
            pic_names = os.listdir(pic_dir)
            record_no = list(map(lambda pic_name: pic_name.split('_')[3], pic_names))
            df_results = pd.DataFrame({'RECORD_NO': record_no, 'POSITIVE_FLAG_PREDICTED': i, 'PIC_NAMES': pic_names})
            df_results.to_csv(result_file, index=False)
        result_df_list.append(df_results)
    results_df_concated = pd.concat(result_df_list, ignore_index=True)
    return results_df_concated

def compair_save_apply(line):
    pic_name = line['PIC_NAMES']
    positive_flag_xlsx = line['POSITIVE_FLAG_XLSX']
    positive_flag_predicted = line['POSITIVE_FLAG_PREDICTED']
    if positive_flag_xlsx == 0 and positive_flag_predicted == 0:
        save_fold = 'TN'
        src_fold = 'normal'
    elif positive_flag_xlsx == 0 and positive_flag_predicted == 1:
        save_fold = 'FP'
        src_fold = 'abnormal'
    elif positive_flag_xlsx == 1 and positive_flag_predicted == 0:
        save_fold = 'FN'
        src_fold = 'normal'
    elif positive_flag_xlsx == 1 and positive_flag_predicted == 1:
        save_fold = 'TP'
        src_fold = 'abnormal'
    save_path = os.path.join(save_dir, save_fold, pic_name)
    src_path = os.path.join(pacs_results_dir, src_fold, 'chest', pic_name)
    print(f'src_path:{src_path},save_path:{save_path}')
    shutil.move(src_path, save_path)

if __name__ == '__main__':
    df_xlsx = read_xlsx_file()
    df_predicted = read_predicted_file()
    results_df_merged = pd.merge(df_predicted, df_xlsx, how='inner', on='RECORD_NO')
    results_df_merged.apply(compair_save_apply, axis=1)