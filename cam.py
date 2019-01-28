import cv2
import numpy as np
import os
import pandas as pd
from configparser import ConfigParser
from generator import AugmentedImageSequence
from models.keras import ModelFactory
from keras import backend as kb
from utility import set_sess_cfg


def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer


def create_cam(df_g, output_dir, image_source_dir, model, generator):
    """
    Create a CAM overlay image for the input image

    :param df_g: pandas.DataFrame, bboxes on the same image
    :param output_dir: str
    :param image_source_dir: str
    :param model: keras model
    :param generator: generator.AugmentedImageSequence
    """
    file_name = df_g["file_name"]
    print(f"process image: {file_name}")

    # draw bbox with labels
    img_ori = cv2.imread(filename=os.path.join(image_source_dir, file_name))

    label = df_g["label"]

    output_path = os.path.join(output_dir, f"{label}.{file_name}")

    img_transformed = generator.load_image(file_name)
    img_transformed = generator.transform_batch_images(img_transformed)

    # CAM overlay
    # Get the 512 input weights to the softmax.
    class_weights = model.layers[-1].get_weights()[0]
    final_conv_layer = get_output_layer(model, "bn")
    get_output = kb.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])
    [conv_outputs, predictions] = get_output([np.array([img_transformed])])
    conv_outputs = conv_outputs[0, :, :, :]

    # Create the class activation map.
    cam = np.zeros(dtype=np.float32, shape=(conv_outputs.shape[:2]))
    for i, w in enumerate(class_weights[:]):
        cam += w * conv_outputs[:, :, i]
    print(f"predictions: {predictions}")
    cam /= np.max(cam)
    cam = cv2.resize(cam, img_ori.shape[:2])
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap[np.where(cam < 0.2)] = 0
    img = heatmap * 0.5 + img_ori

    # add label & rectangle
    # ratio = output dimension / 1024
    ratio = 1
    x1 = int(df_g["x"] * ratio)
    y1 = int(df_g["y"] * ratio)
    x2 = int((df_g["x"] + df_g["w"]) * ratio)
    y2 = int((df_g["y"] + df_g["h"]) * ratio)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(img, text=label+':'+str(predictions), org=(5, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8, color=(0, 0, 255), thickness=1)
    cv2.imwrite(output_path, img)


def main():
    # parser config
    config_file = "./config.ini"
    cp = ConfigParser()
    cp.read(config_file)

    # CAM config
    image_source_dir = cp["CAM"].get("image_source_dir")
    output_dir = cp["CAM"].get("output_dir")
    weight_name = cp["CAM"].get("weight_name")
    base_model_name = cp["CAM"].get("base_model_name")
    bbox_list_file = cp["CAM"].get("bbox_list_file")
    test_csv_file = cp["CAM"].get("test_csv_file")
    image_dimension = cp["CAM"].getint("image_dimension")
    weights_path = os.path.join(output_dir, weight_name)
    grayscale = cp["CAM"].getboolean("grayscale")
    if grayscale:
        channel = 1
    else:
        channel = 3

    print("** load model **")
    model_factory = ModelFactory()
    model = model_factory.get_model(
        model_name=base_model_name,
        use_base_weights=False,
        weights_path=weights_path,
        input_shape=(image_dimension, image_dimension, channel),
        transform_14=False)
    print("read bbox list file")
    df_images = pd.read_csv(bbox_list_file, header=None, skiprows=1)
    df_images.columns = ["file_name", "label", "x", "y", "w", "h"]

    print("create a generator for loading transformed images")
    cam_sequence = AugmentedImageSequence(
        dataset_csv_file=test_csv_file,
        source_image_dir=image_source_dir,
        batch_size=1,
        target_size=(image_dimension, image_dimension),
        augmenter=None,
        steps=1,
        shuffle_on_epoch_end=False,
        grayscale=grayscale,
    )

    image_output_dir = os.path.join(output_dir, "cam")
    if not os.path.isdir(image_output_dir):
        os.makedirs(image_output_dir)

    print("create CAM")
    df_images.apply(
        lambda g: create_cam(
            df_g=g,
            output_dir=image_output_dir,
            image_source_dir=image_source_dir,
            model=model,
            generator=cam_sequence,
        ),
        axis=1,
    )


if __name__ == "__main__":
    set_sess_cfg()
    main()
