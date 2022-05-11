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

from mask_generation \
    import vector_labels_to_masks, train_val_test_split, import_vector_labels
from dataset_analysis \
    import mask_pixel_per_image, class_distribution, visualize_class_distribution
from annotation_experiment_evaluation \
    import evaluate_annotation_experiment, \
    visualize_annotation_experiment_confusion_matrix, \
    visualize_annotation_experiment_box_plot
from model_training \
    import model_training, get_datasets, get_test_dataset
from model_evaluation \
    import create_filter_dataset, evaluate_model_predictions, visualize_prediction_confusion_matrix, \
    visualize_prediction_mean_IoUs_as_box_plots, normalize_confusion_matrix_by_rows, \
    visualize_top_median_bottom_predictions_and_ground_truth, calculate_top_median_bottom_5,\
    df_IoU_from_confusion_matrix, model_load, save_prediction_masks
from utils \
    import get_image_gdf_in_directory, geotif_to_png
from visualization import visualization_annotation_agreement
from pv_potential import pv_potential_analysis

### Define paths
from definitions import \
    DATA_DIR_ANNOTATION_EXPERIMENT, \
    DIR_IMAGES_GEOTIFF, \
    DIR_IMAGES_PNG, \
    DIR_MASKS_SUPERSTRUCTURES, \
    DIR_MASKS_SUPERSTRUCTURES_ANNOTATION_EXPERIMENT, \
    DIR_MASKS_PV_AREAS_ANNOTATION_EXPERIMENT, \
    DIR_MASKS_SEGMENTS, \
    DIR_SEGMENTATION_MODEL_DATA, \
    DIR_MASK_FILES, \
    DIR_PREDICTIONS, \
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


########################################################################################################################
### Import images
########################################################################################################################
# initialize png images, if pngs do not exist.
geotif_to_png(DIR_IMAGES_GEOTIFF, DIR_IMAGES_PNG)

# Get ids of all images in geotiff image folder
image_id_list = [id[:-4] for id in os.listdir(DIR_IMAGES_GEOTIFF) if id[-4:] == '.tif']
gdf_images = get_image_gdf_in_directory(DIR_IMAGES_GEOTIFF)

# import labels from annotation experiment
gdf_test_labels = import_vector_labels(
    FILE_VECTOR_LABELS_ANNOTATION_EXPERIMENT,
    'superstructures',
    LABEL_CLASSES_SUPERSTRUCTURES
)


########################################################################################################################
### 1) Create roof superstructure masks from vector labels
########################################################################################################################
gdf_labels_superstructure = vector_labels_to_masks(
    FILE_VECTOR_LABELS_SUPERSTRUCTURES,
    DIR_MASKS_SUPERSTRUCTURES,
    'superstructures',
    LABEL_CLASSES_SUPERSTRUCTURES,
    gdf_images,
    filter=False
)

train_val_test_split(
    gdf_test_labels,
    gdf_images,
    VAL_DATA_CENTER_POINTS,
    LABEL_CLASSES_SUPERSTRUCTURES,
    DIR_IMAGES_PNG,
    DIR_MASKS_SUPERSTRUCTURES,
    DIR_SEGMENTATION_MODEL_DATA
)


########################################################################################################################
### 2) Create roof segment masks from vector labels
########################################################################################################################
gdf_labels_segments = vector_labels_to_masks(
    FILE_VECTOR_LABELS_SEGMENTS,
    DIR_MASKS_SEGMENTS,
    'segments',
    LABEL_CLASSES_SEGMENTS,
    gdf_images,
    filter=False
)
train_val_test_split(
    gdf_test_labels,
    gdf_images,
    VAL_DATA_CENTER_POINTS,
    LABEL_CLASSES_SEGMENTS,
    DIR_IMAGES_PNG,
    DIR_MASKS_SEGMENTS,
    DIR_SEGMENTATION_MODEL_DATA
)


########################################################################################################################
### 3) Analyze the dataset
########################################################################################################################
# calculate the pixel share of the classes for superstructure and segment dataset
class_share_percent_superstructures = mask_pixel_per_image(
    DIR_MASKS_SUPERSTRUCTURES,
    LABEL_CLASSES_SUPERSTRUCTURES,
    IMAGE_SHAPE
)
class_share_percent_segments = mask_pixel_per_image(
    DIR_MASKS_SEGMENTS,
    LABEL_CLASSES_SEGMENTS,
    IMAGE_SHAPE
)

gdf_labels_superstructure = import_vector_labels(
    FILE_VECTOR_LABELS_SUPERSTRUCTURES,
    'superstructures',
    LABEL_CLASSES_SUPERSTRUCTURES
)
gdf_labels_segments = import_vector_labels(
    FILE_VECTOR_LABELS_SEGMENTS,
    'segments',
    LABEL_CLASSES_SEGMENTS
)

# calculate the number of labels and the labeled area for superstructure and segment dataset
label_class_count_superstructures, label_area_count_superstructures = class_distribution(
    gdf_labels_superstructure, LABEL_CLASSES_SUPERSTRUCTURES
)
label_class_count_segments, label_area_count_segments = class_distribution(
    gdf_labels_segments, LABEL_CLASSES_SEGMENTS
)

visualize_class_distribution(LABEL_CLASSES_SEGMENTS)


########################################################################################################################
### 4) Evaluate annotation experiment and visualize results
########################################################################################################################
# Evaluation annotation agreement of superstructure labels. This takes long time to compute.
# Check if results of evaluation are already saved as pkl file.
# Important: change the pkl filename when evaluating multiple models!
if os.path.isfile('data\\res_annotation_experiment.pkl'):
    with open('data\\res_annotation_experiment.pkl', 'rb') as f:
        [CM_AE_all, CM_AE_list, CM_AE_class_agnostic_all, CM_AE_class_agnostic_list] = pickle.load(f)
    # generate dataframes with all class specific IoUs
    df_IoU_AE = df_IoU_from_confusion_matrix(CM_AE_list, LABEL_CLASSES_SUPERSTRUCTURES)
    df_IoU_AE_class_agnostic = df_IoU_from_confusion_matrix(CM_AE_class_agnostic_list, ['label_class', 'background'])
else:
    image_id_list_annotation_experiment = os.listdir(DIR_MASKS_SUPERSTRUCTURES_ANNOTATION_EXPERIMENT)

    df_IoU_AE, CM_AE_all, CM_AE_list, df_IoU_AE_class_agnostic, CM_AE_class_agnostic_all, CM_AE_class_agnostic_list =\
        evaluate_annotation_experiment(
            LABEL_CLASSES_SUPERSTRUCTURES,
            DIR_MASKS_SUPERSTRUCTURES_ANNOTATION_EXPERIMENT,
            image_id_list_annotation_experiment
        )

# visualize an example of two annotators labels
visualization_annotation_agreement(
    gdf_test_labels,
    LABEL_CLASSES_SUPERSTRUCTURES,
    annotator_ids=[1, 3],
    building_id=[5]
)

# visualize class specific annotation agreement as box plot
visualize_annotation_experiment_box_plot(df_IoU_AE, df_IoU_AE_class_agnostic, LABEL_CLASSES_SUPERSTRUCTURES)

# calculate normalized confusion matrix
CM_AE_all_normalized = normalize_confusion_matrix_by_rows(CM_AE_all)
visualize_annotation_experiment_confusion_matrix(CM_AE_all_normalized, LABEL_CLASSES_SUPERSTRUCTURES.values())

# Evaluation annotation agreement of roof outline. This takes long time to compute
# Check if results of evaluation are already saved as pkl file.
# Important: change the pkl filename when evaluating multiple models!
if os.path.isfile('data\\res_annotation_experiment_pv_areas.pkl'):
    with open('data\\res_annotation_experiment_pv_areas.pkl', 'rb') as f:
        [CM_AE_pv_area_all, CM_AE_pv_area_list] = pickle.load(f)
else:
    image_id_pv_area_list_annotation_experiment = os.listdir(DIR_MASKS_PV_AREAS_ANNOTATION_EXPERIMENT)
    _, _, CM_AE_pv_area_list, _, _, _ =\
        evaluate_annotation_experiment(
            LABEL_CLASSES_PV_AREAS,
            DIR_MASKS_PV_AREAS_ANNOTATION_EXPERIMENT,
            image_id_pv_area_list_annotation_experiment,
        )


# ########################################################################################################################
# ### 5) Train model for semantic segmentation of superstructure - Make sure to use a GPU
# ########################################################################################################################
# model = model_training(MODEL_TYPE, LABEL_CLASSES_SUPERSTRUCTURES, DIR_SEGMENTATION_MODEL_DATA, DIR_RESULTS_TRAINING,
#                        IMAGE_SHAPE)


########################################################################################################################
### 6) Evaluate model and visualize results
########################################################################################################################
DIR_SEGMENTATION_MODEL_DATA = DIR_SEGMENTATION_MODEL_DATA # + '_3' # use validation split 3

# load model and datasets
model, preprocess_input = model_load(MODEL_NAME, MODEL_TYPE, BACKBONE, LABEL_CLASSES_SUPERSTRUCTURES)

train_dataset, valid_dataset, _ = get_datasets(DIR_SEGMENTATION_MODEL_DATA, DIR_MASK_FILES, DATA_VERSION,
                                               preprocess_input, LABEL_CLASSES_SUPERSTRUCTURES.values(), resize=None)

dir_mask_files_test = os.path.join(DIR_SEGMENTATION_MODEL_DATA, 'filenames_annotation_experiment')
test_dataset = get_test_dataset(DATA_DIR_ANNOTATION_EXPERIMENT, dir_mask_files_test, 'annotation_experiment',
                                preprocess_input,
                                LABEL_CLASSES_SUPERSTRUCTURES.values())

filter_dataset = create_filter_dataset(DATA_DIR_ANNOTATION_EXPERIMENT, dir_mask_files_test, 'annotation_experiment',
                                       LABEL_CLASSES_SUPERSTRUCTURES, preprocess_input)

# Evaluation of model.  This takes long time top compute on CPU.
# Check if results of evaluation are already saved as pkl file.
# Important: change the pkl filename when evaluating multiple models!
results_path = os.path.join('data\\res_model_predictions', 'res_model_predictions_UNet_2_initial.pkl')
if os.path.isfile(results_path):
    with open(results_path, 'rb') as f:
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
            filter_center_roof=True
        )

    with open(results_path, 'wb') as f:
        pickle.dump([CM_all, CM_list, CM_class_agnostic_all, CM_class_agnostic_list], f)

# calculate normalized confusion matrix
CM_all_normalized = normalize_confusion_matrix_by_rows(CM_all)
visualize_prediction_confusion_matrix(CM_all_normalized, LABEL_CLASSES_SUPERSTRUCTURES.values())

# visualized the class specific mean IoUs as box plots and add other IoUs as comparison
visualize_prediction_mean_IoUs_as_box_plots(
    df_IoUs,
    df_IoU_class_agnostic,
    df_IoU_AE,
    df_IoU_AE_class_agnostic,
    CM_AE_pv_area_list,
    LABEL_CLASSES_SUPERSTRUCTURES.values()
)

# visualized six images, two good, two medium and two bad predictions
id_top_5, id_median_5, id_bottom_5 = calculate_top_median_bottom_5(CM_list)

visualize_top_median_bottom_predictions_and_ground_truth(
    model,
    id_top_5,
    id_median_5,
    id_bottom_5,
    test_dataset,
    filter_dataset,
    LABEL_CLASSES_SUPERSTRUCTURES
)


########################################################################################################################
### 7) Conduct PV Potential Assessment
########################################################################################################################
# calculate predictions on the validation dataset and save the masks
save_prediction_masks(model, valid_dataset, LABEL_CLASSES_SUPERSTRUCTURES, DIR_PREDICTIONS)
# use prediction masks to calculate pv potential for 6 use cases
pv_potential_analysis()
