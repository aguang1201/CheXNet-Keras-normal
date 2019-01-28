from PIL import Image
import os.path
import glob
import shutil
import pandas as pd


years = list(range(2018, 2019))
months = list(range(1, 4))
days = list(range(1, 32))
input_dir = '/media/room/f7a2c6c3-2763-4a82-b0b1-46378cfbb27e/dataset/pacs_results/compared_with_hospital_results/FN'
output_basedir = '/media/room/COMMON/pacs_dcm'
output_dir = os.path.join(output_basedir, 'FN')
def read_img():
    for year in years:
        year = str(year)
        dcm_dir_year = os.path.join(output_dir, year)
        if not os.path.exists(dcm_dir_year):
            os.makedirs(dcm_dir_year)
        for month in months:
            month = str(month)
            dcm_dir_year_month = os.path.join(dcm_dir_year, month)
            if not os.path.exists(dcm_dir_year_month):
                os.makedirs(dcm_dir_year_month)
            for day in days:
                files_dcm_day = []
                subdirs_dcm_day = []
                day = str(day)
                dcm_dir_year_month_day = os.path.join(dcm_dir_year_month, day)
                if not os.path.exists(dcm_dir_year_month_day):
                    os.makedirs(dcm_dir_year_month_day)
                files_names = glob.glob(os.path.join(input_dir, f'{year}_{month}_{day}_*.png'))
                if len(files_names) == 0:
                    os.rmdir(dcm_dir_year_month_day)
                    continue

                for file_path in files_names:
                    # file_image:2018_3_24_20180324710971_0000001.png
                    file_image = os.path.split(file_path)[1]
                    subdir_dcm = file_image.split('_')[3]
                    file_dcm = file_image.split('_')[4].split('.')[0]
                    dir_dcm = os.path.join(dcm_dir_year_month_day, subdir_dcm)
                    dst = os.path.join(dir_dcm, file_dcm + '.dcm')
                    if os.path.exists(dst):
                        continue
                    if not os.path.exists(dir_dcm):
                        os.makedirs(dir_dcm)
                    src = os.path.join(output_basedir, year, month, day, subdir_dcm, file_dcm + '.dcm')
                    print(f'copy FN dcm: {src} to {dst}')
                    shutil.copy(src, dst)
                    files_dcm_day.append(file_dcm)
                    subdirs_dcm_day.append(subdir_dcm)
                df_dcms = pd.DataFrame({'STUDY_NAME': subdirs_dcm_day, 'DCM_NAME': files_dcm_day, 'LABEL': None})
                df_dcms.sort_values(by=['STUDY_NAME'], inplace=True)
                df_dcms.to_excel(os.path.join(dcm_dir_year_month_day, 'dcm_labels.xls'), index=False)


if __name__ == '__main__':
    read_img()