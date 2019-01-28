import os, shutil
import pandas as pd


xls_dir = "/media/room/f7a2c6c3-2763-4a82-b0b1-46378cfbb27e1/dataset/pacs_results/hospital_results"
no_result = "/home/room/dataset/pacs_png_2014_2018_chest_no_result"
png_dir = "/media/room/f7a2c6c3-2763-4a82-b0b1-46378cfbb27e1/dataset/pacs_png_2014_2018_chest_fromC"
# png_dir = "/home/room/dataset/pacs_png_2014_2018_chest"
normal_dir = os.path.join(png_dir, "0")
abnormal_dir = os.path.join(png_dir, "1")
result_file_name = os.path.join(xls_dir, "2014-2018_results.csv")

def read_xlsx_file(xls_dir):
    if os.path.exists(result_file_name):
        all_results = pd.read_csv(result_file_name)
    else:
        result_df_list = []
        for xls_file in os.listdir(xls_dir):
            if '.xlsx' == os.path.splitext(xls_file)[-1]:
                xls_path = os.path.join(xls_dir, xls_file)
                df_xls = pd.read_excel(xls_path)
                df_xls['POSITIVE_FLAG'] = df_xls['POSITIVE_FLAG'].apply(lambda flag: 0 if flag == '阴性' else 1)
                result_columns = ['RECORD_NO', 'POSITIVE_FLAG']
                result_df = df_xls[result_columns]
                result_df_list.append(result_df)
        all_results = pd.concat(result_df_list)
        all_results.to_csv(result_file_name, index=False)
    return all_results


if __name__ == "__main__":
    if not os.path.exists(normal_dir):
        os.makedirs(normal_dir)
    if not os.path.exists(abnormal_dir):
        os.makedirs(abnormal_dir)
    result_df = read_xlsx_file(xls_dir)
    result_df.set_index(['RECORD_NO'], inplace=True)
    for png in os.listdir(png_dir):
        if '.png' == os.path.splitext(png)[-1]:
            record_no = int(png.split('_')[3])
            src_file = os.path.join(png_dir, png)
            if result_df.loc[record_no, 'POSITIVE_FLAG'] == 1:
                dst_dir = abnormal_dir
            elif result_df.loc[record_no, 'POSITIVE_FLAG'] == 0:
                dst_dir = normal_dir
            else:
                dst_dir = no_result
            print(f'copy {src_file} to {dst_dir}')
            #  shutil.move(src_file, dst_dir)
            if os.path.exists(os.path.join(dst_dir, png)) or os.path.getsize(src_file) == 0:
                continue
            #os.remove(src_file)
            else:
                shutil.move(src_file, dst_dir)
