import cv2
import glob
import numpy as np

imgs = glob.glob("/home/ys1/dataset/CheXpert-v1.0-small/*/*/*/*.jpg")
print(f'images number is : {len(imgs)}')
print('-'*20)
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
total_mean = np.mean(mean_list)/255
total_std = np.mean(stddv_list)/255
print(total_mean)
print(total_std)
with open('total_mean_std.txt', "w") as f:
    f.write(f"total_mean: {total_mean}\n")
    f.write(f"total_std: {total_std}\n")
