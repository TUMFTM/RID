# This code is based on https://github.com/qubvel/segmentation_models
# Copyright 2018, Pavel Yakubovskiy, License: MIT License
# Adaptions made by Sebastian Krapf
__author__ = "Pavel Yakubovskiy"
__copyright__ = "Copyright 2020"
__license__ = "MTI License"
__version__ = "1.0.1"
__maintainer__ = "Pavel Yakubovskiy"
__repository__ = "https://github.com/qubvel/segmentation_models"
__status__ = "alpha"

import os
import cv2
import keras
import numpy as np

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pylab as plt

import segmentation_models as sm
import albumentations as A
from datetime import datetime

from definitions import LABEL_CLASSES_SUPERSTRUCTURES, MODEL_NAME, MODEL_TYPE, DIR_SEGMENTATION_MODEL_DATA, \
    DIR_RESULTS_TRAINING, DATA_VERSION


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# todo: delete these comments
# label_classes_superstructures_annotation_experiment = ['pvmodule', 'dormer', 'window', 'ladder', 'chimney', 'shadow', 'tree', 'unknown'] #'
#
# LABEL_CLASSES_SUPERSTRUCTURES = dict(zip(np.arange(0, len(label_classes_superstructures_annotation_experiment)),
#                                          label_classes_superstructures_annotation_experiment))
#
# DIR_SEGMENTATION_MODEL_DATA = "segmentation_model_data"
# DIR_MASKS = "filenames_initial"
# DIR_RESULTS_TRAINING = "results"
#
# MODEL_NAME = 'UNet'
# MODEL_TYPE = 'UNet' # options are: 'UNet', 'FPN' or 'PSPNet'
# BACKBONE = 'resnet34' #resnet34, efficientnetb2
# DATA_VERSION = '5_initial'  # 2_rev, 3_rev, 4_initial ...
# IMAGE_SHAPE = [512, 512]
#

########################################################################################
########################### Dataloader and utility functions ###########################
########################################################################################
# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    #plt.show()


# helper function for data visualization
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x


# helper function to simplify code
def read_filenames(path):
    with open(path) as f:
        filenames = f.readlines()
    filenames = [filename.replace("\n", "") for filename in filenames]

    return filenames


# classes for data loading and preprocessing
class Dataset:
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    CLASSES = list(LABEL_CLASSES_SUPERSTRUCTURES.values())

    def __init__(
            self,
            images_dir,
            masks_dir,
            filenames,
            classes=None,
            augmentation=None,
            preprocessing=None,
            resize=None
    ):
        self.ids = filenames
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

        self.resize = resize

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        # PSPNet requires a specific image size -> resize image
        if self.resize != None:
            image = cv2.resize(image, [self.resize, self.resize, 3], interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, [self.resize, self.resize, 1], interpolation=cv2.INTER_AREA)
            print(image.size)
            print(mask.size)

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)


class Dataloder(keras.utils.Sequence):
    """Load data from dataset and form batches

    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integer number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):

        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        batch = tuple(batch)

        return batch

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)

########################################################################################
################################### Augmentations ######################################
########################################################################################
def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)


def get_datasets(DIR_SEGMENTATION_MODEL_DATA, DIR_MASK_FILES, DATA_VERSION, preprocess_input, CLASSES, resize=None):

    # directory to images and masks
    x_dir = os.path.join(DIR_SEGMENTATION_MODEL_DATA, 'images')
    y_dir = os.path.join(DIR_SEGMENTATION_MODEL_DATA, 'masks_superstructures_reviewed')

    # filenames of train, validation and test set
    train_filenames = read_filenames(os.path.join(DIR_MASK_FILES , 'train_filenames_' + DATA_VERSION + '.txt'))
    val_filenames = read_filenames(os.path.join(DIR_MASK_FILES, 'val_filenames_' + DATA_VERSION + '.txt'))
    test_filenames = read_filenames(os.path.join(DIR_MASK_FILES, 'test_filenames_' + DATA_VERSION + '.txt'))

    # Dataset for train images
    train_dataset = Dataset(
        x_dir,
        y_dir,
        train_filenames,
        classes=CLASSES,
        preprocessing=get_preprocessing(preprocess_input),
        resize=resize
    )

    # Dataset for validation images
    valid_dataset = Dataset(
        x_dir,
        y_dir,
        val_filenames,
        classes=CLASSES,
        preprocessing=get_preprocessing(preprocess_input),
        resize=resize
    )

    # Dataset for test images
    test_dataset = Dataset(
        x_dir,
        y_dir,
        test_filenames,
        classes=CLASSES,
        preprocessing=get_preprocessing(preprocess_input),
        resize=resize
    )

    return train_dataset, valid_dataset, test_dataset


def get_test_dataset(DIR_SEGMENTATION_MODEL_DATA, DIR_MASK_FILES, DATA_VERSION, preprocess_input, CLASSES, resize=None):
    # directory to images and masks
    x_dir = os.path.join(DIR_SEGMENTATION_MODEL_DATA, 'test')
    y_dir = os.path.join(DIR_SEGMENTATION_MODEL_DATA, 'test_masks')

    # filenames of train, validation and test set
    test_filenames = read_filenames(os.path.join(DIR_MASK_FILES, 'test_filenames_' + DATA_VERSION + '.txt')) #+ '_reviewed'    # + '_reviewed.txt'

    # Dataset for test images
    test_dataset = Dataset(
        x_dir,
        y_dir,
        test_filenames,
        classes=CLASSES,
        preprocessing=get_preprocessing(preprocess_input),
        resize=resize
    )
    return test_dataset


def model_definition(label_classes, model_type, backbone):
    n_classes = len(label_classes) + 1

    BACKBONE = backbone
    LR = 0.0001

    # define network parameters
    activation = 'softmax'

    # create model
    if model_type == 'UNet':
        model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)
    elif model_type == 'FPN':
        model = sm.FPN(BACKBONE, classes=n_classes, activation=activation)
    elif model_type == 'PSPNet':
        model = sm.PSPNet(BACKBONE, classes=n_classes, activation=activation)
    else:
      print('model_type not defined, choose between UNet, FPT or PSPNet')

    preprocess_input = sm.get_preprocessing(BACKBONE)

    # define optomizer
    optim = keras.optimizers.Adam(LR)

    # define loss function and metrics
    total_loss = sm.losses.categorical_focal_jaccard_loss
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

    # compile keras model with defined optimozer, loss and metrics
    model.compile(optim, total_loss, metrics)

    return model, preprocess_input, metrics


def save_training_configuration(dir_results, MODEL_TYPE, BACKBONE, DATA_VERSION, IMAGE_SIZE_X, BATCH_SIZE, EPOCHS, CLASSES, loss, metrics):
    filepath = dir_results + '/results.txt'
    with open(filepath, 'w') as f:
        f.write('model type: ' + MODEL_TYPE + '\n')
        f.write('backbone: ' + BACKBONE + '\n')
        f.write('input data: ' + DATA_VERSION + '\n')
        f.write('image size: ' + str(IMAGE_SIZE_X) + '\n')
        f.write('batch size: ' + str(BATCH_SIZE) + '\n')
        f.write('epochs: ' + str(EPOCHS) + '\n')
        f.write('classes: ' + str(CLASSES) + '\n')
        f.write('loss: ' + str(loss) + '\n')
        f.write('metrics: ' + str(metrics) + '\n')
    return


def model_training(MODEL_TYPE, BACKBONE, LABEL_CLASSES_SUPERSTRUCTURES, DIR_SEGMENTATION_MODEL_DATA, DIR_MASKS,
                   DIR_RESULTS_TRAINING, IMAGE_SHAPE, RESIZE=None):
    ########################################################################################
    ######################### Define classes and dataset directory##########################
    ########################################################################################
    CLASSES = LABEL_CLASSES_SUPERSTRUCTURES.values()

    IMAGE_SIZE_X = IMAGE_SHAPE[0]
    IMAGE_SIZE_Y = IMAGE_SHAPE[1]

    ########################################################################################
    ############################ Segmentation model training ###############################
    ########################################################################################
    BACKBONE = BACKBONE
    BATCH_SIZE = 8
    EPOCHS = 40

    n_classes = len(CLASSES) + 1  # case for binary and multiclass segmentation
    model, preprocess_input, metrics = model_definition(CLASSES, MODEL_TYPE, BACKBONE)

    if MODEL_TYPE == 'PSPNet':
        RESIZE = 480

    train_dataset, valid_dataset, _ = get_datasets(DIR_SEGMENTATION_MODEL_DATA, DIR_MASKS, DATA_VERSION, preprocess_input, CLASSES, resize=RESIZE)

    train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)

    # check shapes for errors
    assert train_dataloader[0][0].shape == (BATCH_SIZE, IMAGE_SIZE_X, IMAGE_SIZE_Y, 3)
    assert train_dataloader[0][1].shape == (BATCH_SIZE, IMAGE_SIZE_X, IMAGE_SIZE_Y, n_classes)

    # define callbacks for learning rate scheduling and best checkpoints saving
    callbacks = [
        keras.callbacks.ModelCheckpoint(DIR_RESULTS_TRAINING + '/' + MODEL_NAME + '.h5',
                                        save_weights_only=True, save_best_only=True, mode='min'),
        keras.callbacks.ReduceLROnPlateau(),
    ]

    # train model
    history = model.fit_generator(
        train_dataloader,
        steps_per_epoch=len(train_dataloader),
        epochs=EPOCHS,
        callbacks=callbacks,
        validation_data=valid_dataloader,
        validation_steps=len(valid_dataloader),
    )

    # Plot training & validation iou_score values
    plt.figure(figsize=(30, 5))
    plt.subplot(121)
    plt.plot(history.history['iou_score'])
    plt.plot(history.history['val_iou_score'])
    plt.title('Model iou_score')
    plt.ylabel('iou_score')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(DIR_RESULTS_TRAINING + '/train_iou_curve')

    # Plot training & validation loss values
    plt.subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(DIR_RESULTS_TRAINING + '/train_loss_curve')

    ########################################################################################
    ################################# Model Evaluation #####################################
    ########################################################################################
    _, _, test_dataset = get_datasets(DIR_SEGMENTATION_MODEL_DATA, DIR_MASKS, DATA_VERSION, preprocess_input, CLASSES, resize=RESIZE)

    test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)

    # load best weights
    model.load_weights(DIR_RESULTS_TRAINING + '/' + MODEL_NAME + '.h5')

    scores = model.evaluate_generator(test_dataloader)

    print("Loss: {:.5}".format(scores[0]))
    for metric, value in zip(metrics, scores[1:]):
       print("mean {}: {:.5}".format(metric.__name__, value))

    save_training_configuration(DIR_RESULTS_TRAINING, MODEL_TYPE, BACKBONE, DATA_VERSION, IMAGE_SIZE_X, BATCH_SIZE, EPOCHS, CLASSES,
                                scores[0], scores[1:])

    return model

# # for debugging Dataloader
# from definitions import LABEL_CLASSES_SUPERSTRUCTURES, DIR_SEGMENTATION_MODEL_DATA, IMAGE_SHAPE
# model_training(LABEL_CLASSES_SUPERSTRUCTURES, DIR_SEGMENTATION_MODEL_DATA, IMAGE_SHAPE)
#
# now = datetime.now()
# res_time_path = str(now.year) + '_' \
#                 + str(now.month) + '_' \
#                 + str(now.day) + '_' \
#                 + str(now.hour) + '_' \
#                 + str(now.minute)
#
# classes = LABEL_CLASSES_SUPERSTRUCTURES
# model_type = MODEL_TYPE
# dir_input_data = DIR_SEGMENTATION_MODEL_DATA
# dir_results = DIR_RESULTS_TRAINING + '_' + res_time_path + '_' + MODEL_NAME + '_' + DATA_VERSION
# os.makedirs(dir_results)
# image_shape = [512, 512]
# dir_masks = DIR_MASKS
#
# model_training(model_type, BACKBONE, classes, dir_input_data, dir_masks, dir_results, image_shape, RESIZE=None)