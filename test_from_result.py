import pandas as pd
import os
import numpy as np
from configparser import ConfigParser


def max_threshold(f1_score_list):
    max_index = np.argmax(f1_score_list)
    return max_index, max(f1_score_list)


# parser config
config_file = "./config.ini"
cp = ConfigParser()
cp.read(config_file)
# default config
class_names = cp["DEFAULT"].get("class_names").split(",")
output_dir = cp["TEST"].get("output_dir")
f1_score_log_path = os.path.join(output_dir, "f1_score.csv")
df_f1_score = pd.read_csv(f1_score_log_path)
f1_score_max_list = df_f1_score.apply(max_threshold)
f1_score_max_path = os.path.join(output_dir, "f1_score_max.csv")
df_f1_score_max = pd.DataFrame(data=list(f1_score_max_list), columns=['threshold_max', 'f1_score_max'])
df_f1_score_max['class_names'] = class_names
df_f1_score_max[['class_names', 'f1_score_max', 'threshold_max']].to_csv(f1_score_max_path, index=False)