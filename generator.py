import numpy as np
import os
import pandas as pd
from keras.utils import Sequence
from PIL import Image
from skimage.transform import resize
from skimage import exposure


class AugmentedImageSequence(Sequence):
    """
    Thread-safe image generator with imgaug support

    For more information of imgaug see: https://github.com/aleju/imgaug
    """

    def __init__(self, dataset_csv_file, source_image_dir, batch_size=16,
                 target_size=(224, 224), augmenter=None, verbose=0, steps=None,
                 shuffle_on_epoch_end=True, random_state=1, grayscale=False):
        """
        :param dataset_csv_file: str, path of dataset csv file
        :param batch_size: int
        :param target_size: tuple(int, int)
        :param augmenter: imgaug object. Do not specify resize in augmenter.
                          It will be done automatically according to input_shape of the model.
        :param verbose: int
        """
        self.dataset_df = pd.read_csv(dataset_csv_file)
        self.source_image_dir = source_image_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.augmenter = augmenter
        self.verbose = verbose
        self.shuffle = shuffle_on_epoch_end
        self.random_state = random_state
        self.prepare_dataset()
        self.grayscale = grayscale
        if steps is None:
            self.steps = int(np.ceil(len(self.x_path) / float(self.batch_size)))
        else:
            self.steps = int(steps)

    def __bool__(self):
        return True

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        batch_x_path = self.x_path[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.asarray([self.load_image(x_path) for x_path in batch_x_path])
        batch_x = self.transform_batch_images(batch_x)
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y

    def crop_image(self, image_array, crop_ratio=0.05):
        image_shape = image_array.shape
        return image_array[int(image_shape[0] * crop_ratio):int(image_shape[0] * (1 - crop_ratio)),
               int(image_shape[1] * crop_ratio):int(image_shape[1] * (1 - crop_ratio))]

    def histogram_equalization(self, image_array):
        return exposure.equalize_hist(image_array)

    def adapthist_equalization(self, image_array):
        return exposure.equalize_adapthist(image_array, clip_limit=0.03)

    def load_image(self, image_file):
        image_path = os.path.join(self.source_image_dir, image_file)
        image = Image.open(image_path)
        if self.grayscale:
            image_array = np.asarray(image.convert('L'))
            image_array = np.expand_dims(image_array, axis=-1)
        else:
            image_array = np.asarray(image.convert("RGB"))
        # image_array = self.crop_image(image_array)
        # image_array = self.histogram_equalization(image_array)
        # image_array = self.adapthist_equalization(image_array)
        image_array = image_array / 255.
        image_array = resize(image_array, self.target_size)
        return image_array

    def transform_batch_images(self, batch_x):
        if self.augmenter is not None:
            batch_x = self.augmenter.augment_images(batch_x)
        if self.grayscale:
            batch_x = batch_x - 0.5
        else:
            # imagenet_mean = np.array([0.485, 0.456, 0.406])
            # imagenet_std = np.array([0.229, 0.224, 0.225])
            imagenet_mean = np.array([0.52, 0.52, 0.52])
            imagenet_std = np.array([0.2, 0.2, 0.2])
            batch_x = (batch_x - imagenet_mean) / imagenet_std
        return batch_x

    def get_y_true(self):
        """
        Use this function to get y_true for predict_generator
        In order to get correct y, you have to set shuffle_on_epoch_end=False.

        """
        if self.shuffle:
            raise ValueError("""
            You're trying run get_y_true() when generator option 'shuffle_on_epoch_end' is True.
            """)
        return self.y[:self.steps*self.batch_size]

    def prepare_dataset(self):
        df = self.dataset_df.sample(frac=1., random_state=self.random_state)
        # self.x_path, self.y = df["Image Index"].as_matrix(), df[self.class_names].as_matrix()
        self.x_path = df["Image Index"].as_matrix()
        self.y = df["Finding Labels"].apply(lambda label: 0 if label == 'No Finding' else 1).as_matrix()

    def on_epoch_end(self):
        if self.shuffle:
            self.random_state += 1
            self.prepare_dataset()
