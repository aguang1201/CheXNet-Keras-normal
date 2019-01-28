import os
from configparser import ConfigParser
import pandas as pd
import shutil

def main():
    # parser config
    config_file = "./config.ini"
    cp = ConfigParser()
    cp.read(config_file)
    class_names = cp["DEFAULT"].get("class_names").split(",")
    dataset_csv_dir = cp["TRAIN"].get("dataset_csv_dir")
    image_source_dir = cp["TRAIN"].get("image_source_dir")
    test_csv = os.path.join(dataset_csv_dir, 'test.csv')
    df_test = pd.read_csv(test_csv)

    for class_name in class_names:
        dst_dir = os.path.join('/data/00_データセット(dataset)/ChestX-ray14/test/' + class_name)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
            image_index_list = df_test[df_test[class_name] == 1]['Image Index']
            for img in image_index_list:
                shutil.copy(os.path.join(image_source_dir, img), dst_dir)

if __name__ == "__main__":
    main()