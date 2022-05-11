__author__ = "Sebastian Krapf"
__copyright__ = "Copyright 2021, "
__credits__ = []
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Sebastian Krapf"
__email__ = "sebastian.krapf@tum.de"
__status__ = "alpha"


import cv2
import numpy as np
from model_evaluation import confusion_matrices, normalize_confusion_matrix_by_rows, df_IoU_from_confusion_matrix
from visualization import box_plot_IoU_TUM_CI, confusion_matrix_heatmap_TUM_CI
from utils import get_progress_string

#############################################################
##### Calculate confusion of GT labels of each labeler ######
#############################################################
def evaluate_annotation_experiment(label_classes, dir_image_annotation_experiment, image_id_list_annotation_experiment):
    labeler_ids = np.arange(0, 5)

    num_images = len(image_id_list_annotation_experiment)/len(labeler_ids) #assumption: each labeler labeled each image

    CM_AE_list = []
    CM_AE_class_agnostic_list = []
    CM_AE_all = np.zeros([len(label_classes)+1, len(label_classes)+1])
    CM_AE_class_agnostic_all = np.zeros([2, 2])

    print('')
    for count, img_id in enumerate(np.array(np.arange(0, num_images), 'int')):

        progress_string = get_progress_string(round(count / len(np.array(np.arange(0, num_images), 'int')), 2))
        print('Evaluating annotation experiment: ' + progress_string, end="\r")

        for labeler_id in labeler_ids:
            gt_image_file = image_id_list_annotation_experiment[img_id * len(labeler_ids) + labeler_id]
            compare_labeler_ids = labeler_ids[labeler_ids!=labeler_id]
            compare_image_files = [image_id_list_annotation_experiment[img_id * len(labeler_ids) + id] for id in compare_labeler_ids]
            gt_img = cv2.imread(dir_image_annotation_experiment + '\\' + gt_image_file, 0)
            compare_images = [cv2.imread(dir_image_annotation_experiment + '\\' + compare_image_file, 0) for compare_image_file in compare_image_files]
            for compare_image in compare_images:
                CM_AE, CM_AE_class_agnostic = confusion_matrices(gt_img, compare_image, label_classes)
                # class sensitive
                CM_AE_all += CM_AE
                CM_AE_list.append(CM_AE)

                # class agnostic
                CM_AE_class_agnostic_all += CM_AE_class_agnostic
                CM_AE_class_agnostic_list.append(CM_AE_class_agnostic)

    # CM_AE_list_normalized = [normalize_confusion_matrix_by_rows(C) for C in CM_AE_list]
    # CM_AE_all_normalized = normalize_confusion_matrix_by_rows(CM_AE_all)

    # CM_AE_list_class_agnostic_normalized = [normalize_confusion_matrix_by_rows(C) for C in CM_AE_class_agnostic_list]
    # CM_AE_class_agnostic_all_normalized = normalize_confusion_matrix_by_rows(CM_AE_class_agnostic_all)

    df_IoU_AE = df_IoU_from_confusion_matrix(CM_AE_list, label_classes)
    df_IoU_AE_class_agnostic = df_IoU_from_confusion_matrix(CM_AE_class_agnostic_list, ['label_class', 'background'])

    return df_IoU_AE, CM_AE_all, CM_AE_list, \
           df_IoU_AE_class_agnostic, CM_AE_class_agnostic_all, CM_AE_class_agnostic_list


def visualize_annotation_experiment_box_plot(df_IoU_AE, df_IoU_AE_class_agnostic, label_classes):
    # boxplot of class specific IoU
    df_AE_lists_notnan = [df_IoU_AE[col][df_IoU_AE[col].notna()] for col in df_IoU_AE]
    df_AE_lists_notnan.append(df_IoU_AE_class_agnostic.label_class)
    class_list = list(label_classes)
    class_list.append('mean IoUs')
    class_list.append('class agnostic RSS')
    box_plot_IoU_TUM_CI(df_AE_lists_notnan, class_list, 'plots/class_IoU_AE.svg', image_size='wide', show_means=True)
    return


def visualize_annotation_experiment_confusion_matrix(CM_AE_all_normalized, class_labels):
    # plot confustion matrix
    data_label_list = list(class_labels)
    data_label_list.append('background')
    save_path = 'plots\\cm_all_AE.svg'
    confusion_matrix_heatmap_TUM_CI(CM_AE_all_normalized, data_label_list, save_path)

    return
