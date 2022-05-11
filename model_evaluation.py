__author__ = "Sebastian Krapf"
__copyright__ = "Copyright 2021, "
__credits__ = []
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Sebastian Krapf"
__email__ = "sebastian.krapf@tum.de"
__status__ = "alpha"

import numpy as np
import os
import sklearn
import cv2

from utils import normalize_confusion_matrix_by_rows, metrics_from_confusion_matrix, df_IoU_from_confusion_matrix, get_progress_string
from visualization import box_plot_IoU_TUM_CI, TUM_CI_colors, confusion_matrix_heatmap_TUM_CI, box_plot_TUM_CI_annotation_and_prediction, visualize_image_gt_pr
from model_training import model_definition, Dataset, get_preprocessing, denormalize, read_filenames

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def model_load(model_name, model_type, backbone, label_classes):
    model, preprocess_input, metrics = model_definition(label_classes, model_type, backbone)
    # load best weights
    model.load_weights(model_name + '.h5')
    return model, preprocess_input


def create_filter_dataset(DATA_DIR_ANNOTATION_EXPERIMENT, DIR_MASK_FILES, DATA_VERSION, label_classes,
                          preprocess_input, resize=None):
    # Create Filter Dataset Based on Roof Outline
    x_filter_dir = os.path.join(DATA_DIR_ANNOTATION_EXPERIMENT, 'test')

    # for reviewed test dataset:
    # x_filter_dir = os.path.join(DATA_DIR_ANNOTATION_EXPERIMENT, 'test_masks_reviewed_filtered', 'test')
    y_filter_dir = os.path.join(DATA_DIR_ANNOTATION_EXPERIMENT, 'test_roof_outlines') #_reviewed

    filter_filenames = read_filenames(os.path.join(DIR_MASK_FILES, 'test_filenames_' + DATA_VERSION + '.txt'))

    # for reviewed test dataset:
    # filter_filenames = read_filenames(os.path.join(
    #     DIR_MASK_FILES + '_reviewed', 'test_filenames_' + DATA_VERSION + '_reviewed.txt'))

    filter_dataset = Dataset(
        x_filter_dir,
        y_filter_dir,
        filter_filenames,
        classes=[label_classes[0]], #work around to only get pixels with value 0
        preprocessing=get_preprocessing(preprocess_input),
        resize=resize
    )
    return filter_dataset


def confusion_matrices(gt_img, pr_img, classes):
    gt_img_vector = np.reshape(gt_img.copy(), len(gt_img) * len(gt_img))
    pr_img_vector = np.reshape(pr_img.copy(), len(pr_img) * len(pr_img))
    CM_class_sensitive = sklearn.metrics.confusion_matrix(gt_img_vector, pr_img_vector, labels=np.arange(len(classes) + 1))

    gt_img_vector[gt_img_vector != (len(classes))] = 0
    gt_img_vector[gt_img_vector == (len(classes))] = 1
    pr_img_vector[pr_img_vector != (len(classes))] = 0
    pr_img_vector[pr_img_vector == (len(classes))] = 1
    CM_class_agnostic = sklearn.metrics.confusion_matrix(gt_img_vector, pr_img_vector, labels=np.arange(2))
    return CM_class_sensitive, CM_class_agnostic


def get_image_gt_and_pr_masks(model, id, test_dataset, filter_dataset, label_classes, filter_to_one_roof_only=True):
    # get mask and prediction
    image, gt_mask = test_dataset[id]

    image = np.expand_dims(image, axis=0)
    pr_mask = model.predict(image)

    # reduce to image of class per pixel
    gt_vector = np.argmax(gt_mask, axis=2)
    pr_vector = np.argmax(pr_mask.squeeze(), axis=2)

    # visualize(image=denormalize(image.squeeze()), gt=gt_vector, pr=pr_vector)

    # apply roof filter to ground truth and prediction by setting all pixels outside of roof area to background
    if filter_to_one_roof_only:
        image_control, filter_mask = filter_dataset[id]
        filter_vector = filter_mask.squeeze()
        gt_vector[filter_vector == 0] = len(label_classes)
        pr_vector[filter_vector == 0] = len(label_classes)
        # visualize(image=denormalize(image.squeeze()), gt=gt_vector, pr=pr_vector)
        if np.sum(image_control-image.squeeze()) > 0:
            print('filter and test data are not loading same image')

    return image, gt_vector, pr_vector


def save_prediction_masks(model, image_dataset, label_classes, DIR_PREDICTIONS):

    for count, id in enumerate(np.arange(len(image_dataset))):
        # make prediction
        _, _, predcition = get_image_gt_and_pr_masks(
            model,
            id,
            image_dataset,
            [],
            label_classes,
            filter_to_one_roof_only=False
        )
        image_name = image_dataset.ids[id]
        # filepath = os.path.join(DIR_PREDICTIONS, image_name)
        # cv2.imwrite(filepath, predcition)
        # # for debugging:
        filepath = os.path.join(DIR_PREDICTIONS + '_visible', image_name)
        cv2.imwrite(filepath, predcition * 255 /(len(label_classes)-1))
    return

#############################################################
############## Prediction and Confusion Matrices ############
#############################################################
def evaluate_model_predictions(model, test_dataset, filter_dataset, label_classes, filter_center_roof=False):
    CM_all = np.zeros([len(label_classes)+1, len(label_classes)+1])
    CM_list = []
    CM_class_agnostic_all = np.zeros([2, 2])
    CM_class_agnostic_list = []

    print('')
    for count, id in enumerate(np.arange(len(test_dataset))):
        progress_string = get_progress_string(round(count / len(np.arange(len(test_dataset))), 2))
        print('Evaluating class distribution: ' + progress_string, end="\r")

        image, gt_vector, pr_vector = get_image_gt_and_pr_masks(model, id, test_dataset, filter_dataset, label_classes, filter_center_roof)

        # calculate multiclass confusion matrix
        CM, CM_class_agnostic = confusion_matrices(gt_vector, pr_vector, label_classes)

        gt_img_vector = np.reshape(gt_vector.copy(), len(gt_vector) * len(gt_vector))
        pr_img_vector = np.reshape(pr_vector.copy(), len(pr_vector) * len(pr_vector))

        CM_all = CM_all + CM
        CM_list.append(CM)

        # class agnostic
        CM_class_agnostic_all = CM_class_agnostic_all + CM_class_agnostic
        CM_class_agnostic_list.append(CM_class_agnostic)

    # # normalize confusion matrizes by row
    # CM_list_normalized = [normalize_confusion_matrix_by_rows(C) for C in CM_list]
    # CM_all_normalized = normalize_confusion_matrix_by_rows(CM_all)
    #
    # # normalize class agnostic confusion matrizes by row
    # CM_class_agnostic_list_normalized = [normalize_confusion_matrix_by_rows(C) for C in CM_class_agnostic_list]
    # CM_class_agnostic_all_normalized = normalize_confusion_matrix_by_rows(CM_class_agnostic_all)

    # generate dataframes with all class specific IoUs
    df_IoUs = df_IoU_from_confusion_matrix(CM_list, label_classes)
    df_IoU_class_agnostic = df_IoU_from_confusion_matrix(CM_class_agnostic_list, ['label_class', 'background'])

    return df_IoUs, CM_all, CM_list, df_IoU_class_agnostic, CM_class_agnostic_all, CM_class_agnostic_list



#############################################################
################ Plot Confusion Matrices ####################
#############################################################
def visualize_prediction_confusion_matrix(CM_all_normalized, label_classes):
    # plot confustion matrix
    data_label_list = list(label_classes)
    data_label_list.append('background')
    save_path = 'plots\\cm_all.svg'
    confusion_matrix_heatmap_TUM_CI(CM_all_normalized, data_label_list, save_path)

    return


# calculate confusion matrix of top 5 images and bottom 5 images
def calculate_top_median_bottom_5(CM_list):
    # calculate metrics such as IoU, Precision, etc. from each confusion matrix
    METRICS = [metrics_from_confusion_matrix(cm) for cm in CM_list]
    mean_IoUs = [m_iou[1] for m_iou in METRICS]

    id_top_5 = np.argsort(mean_IoUs)[-5:]
    id_bottom_5 = np.argsort(mean_IoUs)[:5]
    median_id = int(len(mean_IoUs)/2)
    id_median_5 = np.argsort(mean_IoUs)[np.arange(median_id-2, median_id+3)]

    return id_top_5, id_median_5, id_bottom_5


def visualize_top_and_bottom_CM(CM_list, id_top, id_bottom, label_classes):
    CM_top = np.sum([CM_list[i] for i in id_top], 0)
    CM_top = normalize_confusion_matrix_by_rows(CM_top)
    CM_bottom = np.sum([CM_list[i] for i in id_bottom], 0)
    CM_bottom = normalize_confusion_matrix_by_rows(CM_bottom)

    data_label_list = list(label_classes)
    data_label_list.append('background')

    confusion_matrix_heatmap_TUM_CI(CM_top, data_label_list, 'plots/cm_top.svg')
    confusion_matrix_heatmap_TUM_CI(CM_bottom, data_label_list, 'plots/cm_bottom.svg')
    return

#############################################################
##################### Plot Boxplots #########################
#############################################################
def visualize_prediction_mean_IoUs_as_box_plots(df_IoUs, df_IoU_class_agnostic, df_IoU_AE, df_IoU_AE_class_agnostic, CM_AE_pv_area_list, class_labels):

    df_lists_notnan = [df_IoUs[col][df_IoUs[col].notna()] for col in df_IoUs]
    df_lists_notnan.append(df_IoU_class_agnostic.label_class)

    df_AE_lists_notnan = [df_IoU_AE[col][df_IoU_AE[col].notna()] for col in df_IoU_AE]
    df_AE_lists_notnan.append(df_IoU_AE_class_agnostic.label_class)

    # calculate mean IoU of outlines
    df_IoUs_pv_areas_AE = df_IoU_from_confusion_matrix(CM_AE_pv_area_list, ['roof'])

    ####################################
    ### boxplot of class specific IoU ##
    ####################################
    class_list = list(class_labels)
    class_list.append('mean IoU')
    class_list.append('class agnostic')
    box_plot_IoU_TUM_CI(df_lists_notnan, class_list, 'plots/class_IoU.svg', show_means=True)

    df_combined_list = []
    tick_list = []
    # add roof outline IoUs to plot
    df_combined_list.append(df_IoUs_pv_areas_AE.roof)
    tick_list.append('roof outline')

    for i in np.arange(0, len(df_lists_notnan)):
        df_combined_list.append(df_AE_lists_notnan[i])
        df_combined_list.append(df_lists_notnan[i])
        tick_list.append(class_list[i])
        tick_list.append(class_list[i])

    # define colors of boxes
    colors = []
    colors.append(TUM_CI_colors.white)
    for t in enumerate(tick_list):
        colors.append(TUM_CI_colors.light_gray)
        colors.append(TUM_CI_colors.light_blue)

    box_plot_TUM_CI_annotation_and_prediction(df_IoUs_pv_areas_AE.roof, df_AE_lists_notnan, df_lists_notnan, tick_list,
                                                  'plots/class_IoU_annot_predict.svg', image_size='large', show_means=True)

    ####################################
    # boxplot of mean_IoUs per labeler #
    ####################################
    labeler_ids = np.tile(np.arange(1, 6), 26)
    df_IoUs = df_IoUs.assign(labeler_id=labeler_ids)
    df_lists_labeler_id = [df_IoUs.mean_IoU[df_IoUs.labeler_id == id] for id in df_IoUs.labeler_id.unique()]
    box_plot_IoU_TUM_CI(df_lists_labeler_id, df_IoUs.labeler_id.unique(), 'plots/labeler_IoU.svg', show_means=False)

    ####################################
    ## boxplot of mean_IoUs per image ##
    ####################################
    image_ids = list(np.repeat(np.arange(1, 27), 5))
    df_IoUs = df_IoUs.assign(image_ids=image_ids)
    df_lists_image_id = [df_IoUs.mean_IoU[df_IoUs.image_ids == id] for id in df_IoUs.image_ids.unique()]
    df_lists_image_id.append(df_IoUs.mean_IoU)
    image_id_list = list(df_IoUs.image_ids.unique())
    image_id_list.append('mean_IoU')
    box_plot_IoU_TUM_CI(df_lists_image_id, image_id_list, 'plots/image_IoU.svg', show_means=False)

    return


#############################################################
################ Plot Predictions and Images ################
#############################################################
def visualize_top_median_bottom_predictions_and_ground_truth(model, id_top_5, id_median_5, id_bottom_5, test_dataset, filter_dataset, label_classes):
    label_classes = label_classes.values()

    # plot images, ground truth and predictions
    plot_ids = list(i for i in id_top_5[-2:])
    plot_ids.reverse()
    plot_id_add = list(id_median_5[-2:])
    plot_id_add.reverse()
    plot_ids += plot_id_add
    plot_id_add = list(id_bottom_5[1:-2])
    plot_id_add.reverse()
    plot_ids += plot_id_add

    # [plot_ids.append(i) for i in list(id_median_5[:2])]
    # [plot_ids.append(i) for i in list(id_bottom_5[1:])]

    img_list = []
    gt_list = []
    pr_list = []
    for id in plot_ids:
        # get mask and prediction
        image, gt_vector, pr_vector = get_image_gt_and_pr_masks(model, id, test_dataset, filter_dataset, label_classes, filter_to_one_roof_only=True)
        image = denormalize(image.squeeze())
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_list.append(image)
        gt_list.append(gt_vector)
        pr_list.append(pr_vector)

    save_path = 'plots\\image_predictions.svg'
    visualize_image_gt_pr(img_list, gt_list, pr_list, label_classes, save_path)

    return
