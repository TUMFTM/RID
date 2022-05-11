__author__ = "Sebastian Krapf"
__copyright__ = "Copyright 2021, "
__credits__ = []
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Sebastian Krapf"
__email__ = "sebastian.krapf@tum.de"
__status__ = "alpha"


import cv2
import os
import pickle
import sys

import numpy as np
from shapely.geometry import Point

##################################################
################ Define paths ####################
##################################################
# base directory
DIR_BASE = os.path.dirname(sys.argv[0])

# data directory
DIR_DATA = DIR_BASE + "\\data"

# training files directories
DIR_SEGMENTATION_MODEL_DATA = DIR_BASE + "\\" + "segmentation_model_data"

# result directories
DIR_RESULTS_TRAINING = DIR_BASE + "\\results"
DIR_PREDICTIONS = DIR_BASE + '\\predictions'

# make paths if they do not exist
if not os.path.isdir(DIR_DATA): os.mkdir(DIR_DATA)
if not os.path.isdir(DIR_SEGMENTATION_MODEL_DATA): os.mkdir(DIR_SEGMENTATION_MODEL_DATA)
if not os.path.isdir(DIR_RESULTS_TRAINING): os.mkdir(DIR_RESULTS_TRAINING)
if not os.path.isdir(DIR_PREDICTIONS): os.mkdir(DIR_PREDICTIONS)
if not os.path.isdir(DIR_BASE + "\\plot"): os.mkdir(DIR_BASE + "\\plot")

# image directories
DIR_IMAGES_GEOTIFF = DIR_DATA + "\\images_roof_centered_geotiff"  # "images_annotation_experiment_geotiff" #
DIR_IMAGES_PNG = DIR_DATA + "\\images_roof_centered_png"  # images_annotation_experiment_png"
# mask directories
DIR_MASKS_SUPERSTRUCTURES = DIR_DATA + "\\masks_superstructures_reviewed" #_initial"
DIR_MASKS_SEGMENTS = DIR_DATA + "\\masks_segments"
# annotation experiment directories
DATA_DIR_ANNOTATION_EXPERIMENT = DIR_BASE + '\\raster_data_annotation_experiment'
DIR_MASKS_SUPERSTRUCTURES_ANNOTATION_EXPERIMENT = DIR_DATA + "\\masks_superstructures_annotation_experiment"
DIR_MASKS_PV_AREAS_ANNOTATION_EXPERIMENT = DIR_DATA + "\\masks_pv_areas_annotation_experiment"
# training files
DIR_MASK_FILES = DIR_SEGMENTATION_MODEL_DATA + "\\filenames_reviewed"
# vector label files
FILE_VECTOR_LABELS_SUPERSTRUCTURES = "data\\" + "obstacles_reviewed.csv" #_initial.csv" #
FILE_VECTOR_LABELS_SEGMENTS = "data\\" + "segments_reviewed.csv" #_initial.csv" #
FILE_VECTOR_LABELS_PV_AREAS = "data\\" + "pv_areas_reviewed.csv" #_initial.csv" #
FILE_VECTOR_LABELS_ANNOTATION_EXPERIMENT = "data\\" + "obstacles_annotation_experiment.csv"

##################################################
########### Define class definition ##############
##################################################
# ## ALL labeled classes - Choose class definition
# label_classes_superstructures_all = ['background', 'unknown', 'window', 'ladder', 'shadow', 'chimney',
#                      'pvmodule', 'tree', 'dormer', 'balkony']
# label classes used in annotation experiment
label_classes_superstructures_annotation_experiment = ['pvmodule', 'dormer', 'window', 'ladder', 'chimney', 'shadow',
                                                       'tree', 'unknown'] #

## Label classes of segments - Choose class definition
label_classes_segments_6 = ['N', 'E', 'S', 'W', 'flat']
label_classes_segments_10 = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'flat']
label_classes_segments_18 = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                             'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW', 'flat']

label_clases_pv_areas = ['pv_area']

# decide here which classes to use to prepare the dataset!
LABEL_CLASSES_SUPERSTRUCTURES = dict(zip(np.arange(0, len(label_classes_superstructures_annotation_experiment)),
                                         label_classes_superstructures_annotation_experiment))

LABEL_CLASSES_SEGMENTS = dict(zip(np.arange(0, len(label_classes_segments_18)), label_classes_segments_18))

LABEL_CLASSES_PV_AREAS = dict(zip(np.arange(0, len(label_clases_pv_areas)), label_clases_pv_areas))

# Manually define center points of validation data circles
north = Point([11.985659535136675, 48.41290587924208])
west = Point([11.975791189473085, 48.400038828407816])
east = Point([11.99794882677886, 48.39994299101589])
center_north = Point([11.987591755393657, 48.406794909515014])
center_south = Point([11.991633539397318, 48.40346489660213])

VAL_DATA_CENTER_POINTS = list([north, west, east, center_north, center_south])

# Coordinate systems
EPSG_METRIC = 25832

# Neural Network Parameters
MODEL_NAME = 'UNet_2_initial'
MODEL_TYPE = 'UNet' # options are: 'Unet', 'FPN' or 'PSPNet'
BACKBONE = 'resnet34' #resnet34, efficientnetb2
DATA_VERSION = '2_initial'  # 2_rev, 3_rev, 4_initial ...

IMAGE_SHAPE = cv2.imread(DIR_IMAGES_GEOTIFF + '\\' + os.listdir(DIR_IMAGES_GEOTIFF)[0], 0).shape

############################################################
########### Look Up Table Technical Potential ##############
############################################################
lookup_path = os.path.abspath(os.path.join(DIR_BASE, 'data', 'df_technical_potential_lookup.pkl'))
# open lookup table
with open(lookup_path, 'rb') as f:
    df_technical_potential_LUT = pickle.load(f)