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

import numpy as np

from utils import convert_latlon_to_lonlat, convert_lonlat_to_metric, get_progress_string
from visualization import segment_polar_plot
from definitions import EPSG_METRIC, label_classes_segments_6, label_classes_segments_10, label_classes_segments_18, \
    FILE_VECTOR_LABELS_SEGMENTS
from mask_generation import import_vector_labels


#####################################################
def mask_pixel_per_image(dir_masks, classes, image_shape):
    x_px = image_shape[0]
    y_px = image_shape[1]
    # # superstructure vs background amount of pixels
    file_list = os.listdir(dir_masks)
    BG_count = 0
    class_count = 0
    print('')
    for count, file in enumerate(file_list):
        progress_string = get_progress_string(round(count / len(file_list), 2))
        print('Evaluating class pixels per image: ' + progress_string, end="\r")
        file_path = dir_masks + '\\' + file
        img = cv2.imread(file_path, 0)
        BG_count += np.sum([img == len(classes)]) / (x_px*y_px)
        class_count += np.sum([img != len(classes)]) / (x_px*y_px)

    class_share_percent = class_count/(BG_count + class_count)
    return class_share_percent


def class_distribution(gdf_labels, label_classes):
    if gdf_labels.crs != EPSG_METRIC:
        gdf_labels = gdf_labels.to_crs(EPSG_METRIC)

    label_count = np.array([len(geom) for geom in gdf_labels.geometry])
    label_area = np.array([geom.area for geom in gdf_labels.geometry])
    label_classes_in_gdf = [cl for cl in gdf_labels.class_type]

    label_class_count = np.zeros(len(label_classes))
    label_area_count = label_class_count.copy()

    print('')
    for count, label in enumerate(label_classes.items()):
        progress_string = get_progress_string(round(count / len(label_classes.items()), 2))
        print('Evaluating class distribution: ' + progress_string, end="\r")

        lab_binary = [cl==label[1] for cl in label_classes_in_gdf]
        label_class_count[label[0]] = sum(lab_binary * label_count)
        label_area_count[label[0]] = sum(lab_binary * label_area)
    return label_class_count, label_area_count


def visualize_class_distribution(LABEL_CLASSES_SEGMENTS):
    gdf_labels_segments_6 = import_vector_labels(
        FILE_VECTOR_LABELS_SEGMENTS,
        'segments',
        dict(zip(np.arange(0, len(label_classes_segments_6)), label_classes_segments_6))
    )
    gdf_labels_segments_10 = import_vector_labels(
        FILE_VECTOR_LABELS_SEGMENTS,
        'segments',
        dict(zip(np.arange(0, len(label_classes_segments_10)), label_classes_segments_10))
    )
    gdf_labels_segments_18 = import_vector_labels(
        FILE_VECTOR_LABELS_SEGMENTS,
        'segments',
        dict(zip(np.arange(0, len(label_classes_segments_18)), label_classes_segments_18))
    )

    _, label_area_count_4 = class_distribution(gdf_labels_segments_6, dict(zip(np.arange(0, len(label_classes_segments_6)), label_classes_segments_6)))
    _, label_area_count_8 = class_distribution(gdf_labels_segments_10, dict(zip(np.arange(0, len(label_classes_segments_10)), label_classes_segments_10)))
    _, label_area_count_16 = class_distribution(gdf_labels_segments_18, dict(zip(np.arange(0, len(label_classes_segments_18)), label_classes_segments_18)))

    segment_polar_plot(LABEL_CLASSES_SEGMENTS, label_area_count_4, label_area_count_8, label_area_count_16)

    return