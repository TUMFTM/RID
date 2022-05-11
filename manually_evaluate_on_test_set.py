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
import numpy as np

from model_training \
    import model_training, get_datasets, get_test_dataset, Dataloder, model_definition
from model_evaluation import model_load, save_prediction_masks, create_filter_dataset, df_IoU_from_confusion_matrix, \
    evaluate_model_predictions, normalize_confusion_matrix_by_rows, visualize_prediction_confusion_matrix


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


def find_treshold(label_classes, model_type, backbone, image, gt_mask):
    IoU_list = []
    for th in np.arange(0, 1, 0.1):
        model, _, _ = model_definition(label_classes, model_type, backbone, treshhold=th)
        loss, IoU, F1 = model.evaluate(np.expand_dims(image, axis=0), np.expand_dims(gt_mask, axis=0))
        IoU_list.append(IoU)

    return



MODEL_NAME = 'UNet_2_initial'

DIR_SEGMENTATION_MODEL_DATA = DIR_SEGMENTATION_MODEL_DATA # + '_3' # use validation split 3

# load model and datasets
model, preprocess_input = model_load(MODEL_NAME, MODEL_TYPE, BACKBONE, LABEL_CLASSES_SUPERSTRUCTURES)

# train_dataset, valid_dataset, _ = get_datasets(DIR_SEGMENTATION_MODEL_DATA, DIR_MASK_FILES, DATA_VERSION, preprocess_input,
#                                                           LABEL_CLASSES_SUPERSTRUCTURES.values(), resize=None)



# # on annotation experiment GT dataset
DATA_VERSION = 'annotation_experiment'
DIR_MASK_FILES = os.path.join(DIR_SEGMENTATION_MODEL_DATA, 'filenames_' + DATA_VERSION)
# DATA_DIR_ANNOTATION_EXPERIMENT_test = os.path.join(DATA_DIR_ANNOTATION_EXPERIMENT, 'test_masks_reviewed_filtered') #
# test_dataset = get_test_dataset(DATA_DIR_ANNOTATION_EXPERIMENT, DIR_MASK_FILES, DATA_VERSION, preprocess_input,
#                                 LABEL_CLASSES_SUPERSTRUCTURES.values())
# on normal GT dataset
test_dataset = get_test_dataset(DIR_SEGMENTATION_MODEL_DATA, DIR_MASK_FILES, DATA_VERSION, preprocess_input,
                                LABEL_CLASSES_SUPERSTRUCTURES.values())

filter_dataset = create_filter_dataset(DATA_DIR_ANNOTATION_EXPERIMENT, DIR_MASK_FILES, DATA_VERSION,
                                       LABEL_CLASSES_SUPERSTRUCTURES, preprocess_input)

# Evaluation of model.  This takes long time top compute on CPU.
# Check if results of evaluation are already saved as pkl file.
# Important: change the pkl filename when evaluating multiple models!#
pkl_path = 'data\\res_model_predictions_UNet_2_initial_on_rev_filter_2_TEST.pkl'

if os.path.isfile(pkl_path):
    with open(pkl_path, 'rb') as f:
        [df_IoUs, CM_all, CM_list, CM_class_agnostic_all, CM_class_agnostic_list] = pickle.load(f)
    # generate dataframes with all class specific IoUs
    df_IoUs = df_IoU_from_confusion_matrix(CM_list, LABEL_CLASSES_SUPERSTRUCTURES)
    df_IoU_class_agnostic = df_IoU_from_confusion_matrix(CM_class_agnostic_list, ['label_class', 'background'])
else:
    df_IoUs, CM_all, CM_list, df_IoU_class_agnostic, CM_class_agnostic_all, CM_class_agnostic_list = \
        evaluate_model_predictions(
            model,
            test_dataset,
            filter_dataset,
            LABEL_CLASSES_SUPERSTRUCTURES,
            filter_center_roof=False
        )

    with open(pkl_path, 'wb') as f:
        pickle.dump([df_IoUs, CM_all, CM_list, CM_class_agnostic_all, CM_class_agnostic_list], f)

# score = []
# for i in np.arange(0, 4):
#     score.append(model.evaluate(np.expand_dims(test_dataset[i][0], axis=0), np.expand_dims(test_dataset[i][1],axis=0)))

# calculate normalized confusion matrix
CM_all_normalized = normalize_confusion_matrix_by_rows(CM_all)
visualize_prediction_confusion_matrix(CM_all_normalized, LABEL_CLASSES_SUPERSTRUCTURES.values())


