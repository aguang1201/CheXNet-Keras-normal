import pydicom
from PIL.Image import fromarray
import numpy as np
import os,cv2
import SimpleITK as sitk
from skimage.transform import resize
import io
import PIL.Image as Image
from keras.preprocessing.image import img_to_array


def rescole(img):
    img_score = np.array([np.min(img), np.max(img)])
    # lung_score = np.array([-1200., 600.])
    newimg = (img - img_score[0]) / (img_score[1] - img_score[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    newimg = (newimg * 255).astype('uint8')
    return newimg


def limitedEqualize(img_array, limit=4.0):

    img_array_list = []

    for img in img_array:
        clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=(8, 8))

        img_array_list.append(clahe.apply(img))

    img_array_limited_equalized = np.array(img_array_list)

    return img_array_limited_equalized


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y

def convert_from_dicom_to_png(img, low, high, out_put_size=None):
    newimg = (img-low)/(high-low)    #归一化
    shape = newimg.shape
    print(f'shape is : {shape}')
    if len(shape) == 3:
        new_shape = (shape[1], shape[2], shape[0])
    else:
        new_shape = (shape[1], shape[2], shape[3])
    try:
        newimg = np.reshape(newimg, new_shape)
        if out_put_size:
            img_file = (resize(newimg, (out_put_size, out_put_size, new_shape[2])) * 255).astype('uint8')
        else:
            img_file = (newimg*255).astype('uint8')
        newimg = resize(newimg, (224, 224, new_shape[2]))
        # newimg = np.expand_dims(newimg, axis=-1)
        if new_shape[2] == 1:
            newimg = np.repeat(newimg, 3, axis=2)
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        newimg = (newimg - imagenet_mean) / imagenet_std
        # newimg = prewhiten(newimg)
    except Exception:
        print('reshape or resize Exception')
        pass
    return np.expand_dims(newimg, axis=0), img_file

def convert_from_dicom_to_png_fromC(img, out_put_size=None):
    shape = img.shape
    try:
        img = img/255.
        if out_put_size:
            img_file = (resize(img, (out_put_size, out_put_size)) * 255).astype('uint8')
        newimg = resize(img, (224, 224))
        if shape[2] == 1:
            newimg = np.repeat(newimg, 3, axis=2)
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        newimg = (newimg - imagenet_mean) / imagenet_std
    except Exception:
        print('reshape or resize Exception')
        pass
    return np.expand_dims(newimg, axis=0), img_file

# pydicom or dcmj2pnm
def read_dcm(src_path, dst_path):
    os.system('dcmdjpeg ' + src_path + ' ' + src_path)
    dcm = pydicom.dcmread(src_path)

    try:
        dcm_image = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
        im = fromarray(rescole(dcm_image))
        im.save(dst_path)
        print("pydicom!")
    except Exception:
        os.system('dcmj2pnm +on ' + src_path + ' ' + dst_path)
        print("dcmj2pnm!")
    else:
        pass

# C++
def read_dcm_fromC(src_path, out_put_size=None):
    print(f'image is:{src_path}')
    file_name_list = src_path.split('/')
    # file_name = file_name_list[-2] + file_name_list[-1]
    # file_temp = os.path.join('temp', file_name)
    file_temp = os.path.join('temp', file_name_list[-2] + '_' + file_name_list[-1].split('.')[0] + '.png')
    try:
        os.system('dcmj2pnm +on ' + src_path + ' ' + file_temp)
        with open(file_temp, 'rb') as temp_png:
            img = Image.open(io.BytesIO(temp_png.read()))
            img_array = img_to_array(img)
        os.remove(file_temp)
        if img_array.shape[2] not in [1, 3]:
            return None, None
        return convert_from_dicom_to_png_fromC(img_array, out_put_size=out_put_size)  # 调用函数，转换成jpg文件并保存到对应的路径
    except Exception:
        print(f'{src_path} can not be read by cpp')
        return None, None

# SimpleITK
def read_dcm(src_path, out_put_size=None):
    print(f'image is:{src_path}')
    try:
        img = sitk.ReadImage(src_path)  # 读取dicom文件的相关信息
        img_array = sitk.GetArrayFromImage(img)  # 获取array
        if img_array.shape[0] not in [1, 3]:
            return None, None
        high = np.max(img_array)
        low = np.min(img_array)
        return convert_from_dicom_to_png(img_array, low, high, out_put_size=out_put_size)  # 调用函数，转换成jpg文件并保存到对应的路径
    except Exception:
        print(f'{src_path} can not be read by simpleITK')
        return None, None

# mudicom
# def read_dcm_mudicom(dcm_path):
#     mu = mudicom.load(dcm_path)
#     img = mu.image
#     img_array = img.numpy
#     newimg = resize(newimg, (224, 224, new_shape[2]))
#     # newimg = np.expand_dims(newimg, axis=-1)
#     if new_shape[2] == 1:
#         newimg = np.repeat(newimg, 3, axis=2)
#     imagenet_mean = np.array([0.485, 0.456, 0.406])
#     imagenet_std = np.array([0.229, 0.224, 0.225])
#     newimg = (newimg - imagenet_mean) / imagenet_std
#     return np.expand_dims(newimg, axis=0), img_file


