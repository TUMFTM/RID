__author__ = "Sebastian Krapf"
__copyright__ = "Copyright 2021, "
__credits__ = []
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Sebastian Krapf"
__email__ = "sebastian.krapf@tum.de"
__status__ = "alpha"

import os
import pickle

from model_training \
    import model_training, get_datasets, get_test_dataset
from model_evaluation import model_load, save_prediction_masks


### Define paths
from definitions import \
    DIR_BASE, \
    DATA_DIR_ANNOTATION_EXPERIMENT, \
    DIR_IMAGES_GEOTIFF, \
    DIR_IMAGES_PNG, \
    DIR_MASKS_SUPERSTRUCTURES, \
    DIR_MASKS_SUPERSTRUCTURES_ANNOTATION_EXPERIMENT, \
    DIR_MASKS_PV_AREAS_ANNOTATION_EXPERIMENT, \
    DIR_MASKS_SEGMENTS, \
    DIR_SEGMENTATION_MODEL_DATA, \
    DIR_MASK_FILES, \
    DIR_RESULTS_TRAINING, \
    FILE_VECTOR_LABELS_SUPERSTRUCTURES, \
    FILE_VECTOR_LABELS_SEGMENTS, \
    FILE_VECTOR_LABELS_ANNOTATION_EXPERIMENT, \
    VAL_DATA_CENTER_POINTS,\
    IMAGE_SHAPE, \
    MODEL_NAME, \
    MODEL_TYPE, \
    DATA_VERSION, \
    BACKBONE

### Define labeling classes
from definitions import \
    LABEL_CLASSES_SUPERSTRUCTURES,\
    LABEL_CLASSES_SEGMENTS, \
    LABEL_CLASSES_PV_AREAS

model, preprocess_input = model_load(MODEL_NAME, MODEL_TYPE, BACKBONE, LABEL_CLASSES_SUPERSTRUCTURES)

# train_dataset, valid_dataset, _ = get_datasets(
#     DIR_SEGMENTATION_MODEL_DATA,
#     DIR_MASK_FILES,
#     DATA_VERSION,
#     preprocess_input,
#     LABEL_CLASSES_SUPERSTRUCTURES.values(),
#     resize=None
# )

dir_mask_files_test = os.path.join(DIR_SEGMENTATION_MODEL_DATA, 'filenames_annotation_experiment')
test_dataset = get_test_dataset(DATA_DIR_ANNOTATION_EXPERIMENT, dir_mask_files_test, 'annotation_experiment',
                                preprocess_input,
                                LABEL_CLASSES_SUPERSTRUCTURES.values())

DIR_PREDICTIONS = DIR_BASE + '\\predictions_test'
save_prediction_masks(model, test_dataset, LABEL_CLASSES_SUPERSTRUCTURES, DIR_PREDICTIONS)
