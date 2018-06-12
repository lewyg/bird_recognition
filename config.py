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





# Perceptron
DATA_PATH = os.path.join(RESOURCES_PATH, 'data')
LABELS_PATH = os.path.join(RESOURCES_PATH, 'labels')
HIDDEN_LAYER_SIZES = (
    (420,),
    (140, 280),
    (60, 120, 240),
    (28, 56, 112, 224)
)

PERCEPTRON_MAX_EPOCHS = 220


# VGG16
TOP_MODEL_WEIGHTS_PATH = os.path.join(RESOURCES_PATH, 'top_model2')
BOTTLENECK_TRAIN_FEATURES_PATH = os.path.join(RESOURCES_PATH, 'bottleneck_train.npy')
BOTTLENECK_TEST_FEATURES_PATH = os.path.join(RESOURCES_PATH, 'bottleneck_test.npy')
GROUND_TRUTH_PATH = os.path.join(RESOURCES_PATH, 'ground_truth')


# CNN
SEED = 7
TRAIN_SPLIT_RATIO = 0.833
TEST_SPLIT_RATIO = 0.167
EXAMPLES = 3000
CLASSES = 50
TRAIN_EXAMPLES = EXAMPLES * TRAIN_SPLIT_RATIO
TEST_EXAMPLES = EXAMPLES * TEST_SPLIT_RATIO
BATCH_SIZE = 30
VGG_EPOCHS = 5
TOP_MODEL_MAX_EPOCHS = 300


