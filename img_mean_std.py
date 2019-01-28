import cv2
import glob
import numpy as np

imgs = glob.glob("/home/room/dataset/pacs_png_2014_2018_chest_fromC/1/*.png")
mean_list = []
stddv_list = []
count = 0
for img in imgs:
    count += 1
    im = cv2.imread(img, 0)
    (mean, stddv) = cv2.meanStdDev(im)
    mean_list.append(mean)
    stddv_list.append(stddv)
    if count % 100 == 0:
        print(count)
        print(np.mean(mean_list)/255)
        print(np.mean(stddv_list)/255)
        print('-'*20)