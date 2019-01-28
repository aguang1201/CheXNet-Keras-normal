from imgaug import augmenters as iaa

augmenter = iaa.Sequential(
     [
         iaa.Fliplr(0.5),
     ],
     random_order=True,
)
# augmenter = iaa.Sequential([
#   iaa.Fliplr(0.5),  # horizontal flips
#   iaa.Crop(percent=(0, 0.05)),  # random crops
# ], random_order=True)
