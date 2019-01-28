from PIL import Image
import os.path
import glob
import shutil


folds = ['0_normal', '1_abnormal']

def convert_size(image_path, outdir, width=1024, height=1024):
    img = Image.open(image_path)
    print(f'image_path:{image_path}')
    try:
        new_img = img.resize((width, height), Image.BILINEAR)
        new_img.save(os.path.join(outdir, os.path.basename(image_path)))
    except Exception as e:
        print(e)

def read_img():
    input_dir = '/media/room/COMMON/dataset/pacs_201609_201803'
    output_basedir = '/home/room/dataset/pacs'
    for fold in folds:
        input_images = os.path.join(input_dir, fold, '*.png')
        output_dir = os.path.join(output_basedir, fold)
        images_list = glob.glob(input_images)
        for image in images_list:
            convert_size(image, output_dir)

def clean_img():
    input_dir = '/home/room/dataset/pacs'
    output_basedir = '/media/room/COMMON/False_chest'
    for fold in folds:
        input_images = os.path.join(input_dir, fold, '*.png')
        output_dir = os.path.join(output_basedir, fold)
        images_list = glob.glob(input_images)
        for image in images_list:
            if os.path.splitext(image)[0][-1] != '1':
                print(f'move image:{image} to {output_dir}')
                dst_path = os.path.join(output_dir, os.path.basename(image))
                shutil.move(image, dst_path)

if __name__ == '__main__':
    # read_img()
    clean_img()
