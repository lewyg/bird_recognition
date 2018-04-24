import os

import cv2

BASE_PATH = os.path.abspath(os.path.join(os.curdir, os.pardir)).replace('\\', '/')
RESOURCES_PATH = os.path.join(BASE_PATH, 'resources').replace('\\', '/')
IN_PATH = os.path.join(RESOURCES_PATH, 'in').replace('\\', '/')
OUT_PATH = os.path.join(RESOURCES_PATH, 'out').replace('\\', '/')
BOUNDING_BOXES_PATH = os.path.join(RESOURCES_PATH, 'bounding_boxes').replace('\\', '/')

IMAGE_FORMAT = 'jpg'
IMAGE_SIZE = 224
IMAGE_BORDER_STYLE = cv2.BORDER_DEFAULT  # reflect image on borders

LBP_POINTS_NUMBER = 24
LBP_RADIUS = 8