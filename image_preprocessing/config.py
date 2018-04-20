import os

BASE_PATH = os.path.abspath(os.path.join(os.curdir, os.pardir)).replace('\\', '/')
RESOURCES_PATH = os.path.join(BASE_PATH, 'resources').replace('\\', '/')
IN_PATH = os.path.join(RESOURCES_PATH, 'in').replace('\\', '/')
OUT_PATH = os.path.join(RESOURCES_PATH, 'out').replace('\\', '/')
BOUNDING_BOXES_PATH = os.path.join(RESOURCES_PATH, 'bounding_boxes').replace('\\', '/')

IMAGE_FORMAT = 'jpg'
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
