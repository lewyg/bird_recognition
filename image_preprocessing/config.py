import os

BASE_PATH = os.path.abspath(os.path.join(os.curdir, os.pardir))
RESOURCES_PATH = os.path.join(BASE_PATH, 'resources')
IN_PATH = os.path.join(RESOURCES_PATH, 'in')
OUT_PATH = os.path.join(RESOURCES_PATH, 'out')
BOUNDING_BOXES_PATH = os.path.join(RESOURCES_PATH, 'bounding_boxes')

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
