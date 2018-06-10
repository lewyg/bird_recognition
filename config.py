import os

import cv2
import numpy as np

np.random.seed(seed=1)

BASE_PATH = os.path.abspath(os.path.join(os.curdir, os.pardir)).replace('\\', '/')
RESOURCES_PATH = os.path.join(BASE_PATH, 'resources').replace('\\', '/')
IN_PATH = os.path.join(RESOURCES_PATH, 'in').replace('\\', '/')
OUT_PATH = os.path.join(RESOURCES_PATH, 'out').replace('\\', '/')
PLOT_PATH = os.path.join(RESOURCES_PATH, 'plots').replace('\\', '/')
BOUNDING_BOXES_PATH = os.path.join(RESOURCES_PATH, 'bounding_boxes').replace('\\', '/')
VGG16_WEIGHTS_PATH = os.path.join(RESOURCES_PATH, 'vgg16_weights.h5').replace('\\', '/')

IMAGE_FORMAT = 'jpg'
IMAGE_SIZE = 224
IMAGE_BORDER_STYLE = cv2.BORDER_DEFAULT  # reflect image on borders


LBP_RADIUS = 1
LBP_POINTS_NUMBER = LBP_RADIUS * 8
LBP_METHOD = 'uniform'
LBP_PARTS = 4
LBP_PART_SIZE = int(IMAGE_SIZE / LBP_PARTS)

DATA_PATH = os.path.join(RESOURCES_PATH, 'data')
LABELS_PATH = os.path.join(RESOURCES_PATH, 'labels')
TOP_MODEL_WEIGHTS_PATH = os.path.join(RESOURCES_PATH, 'top_model')
BOTTLENECK_TRAIN_FEATURES_PATH = os.path.join(RESOURCES_PATH, 'bottleneck_train')
BOTTLENECK_TEST_FEATURES_PATH = os.path.join(RESOURCES_PATH, 'bottleneck_test')
SEED = 7

HIDDEN_LAYER_SIZES = (
    (420,),
    (140, 280),
    (60, 120, 240),
    (28, 56, 112, 224)
)

MAX_EPOCHS = 220
