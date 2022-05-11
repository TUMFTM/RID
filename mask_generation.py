__author__ = "Sebastian Krapf"
__copyright__ = "Copyright 2021, "
__credits__ = []
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Sebastian Krapf"
__email__ = "sebastian.krapf@tum.de"
__status__ = "alpha"

from osgeo import gdal
import geopandas as gpd
import pandas as pd
import shapely
from shapely.geometry import MultiPolygon, Polygon, Point, box
shapely.speedups.disable()
import numpy as np
import cv2
import os
import shutil
from utils import convert_between_latlon_and_pixel, wkt_to_shape, azimuth_to_label_class, \
    switch_coordinates, get_progress_string
import pickle


def import_vector_labels(file_vector_labels, mask_generation_case, label_classes):
    label_classes = list(label_classes.values()) #take only label values

    # Import from csv
    df_labels = pd.read_csv(file_vector_labels)

    # Convert geometries in WKT format to shape format and create geodataframe
    # Label classes for superstructures and segments need to be defined in different ways
    if mask_generation_case == 'superstructures':
        df_labels = df_labels.rename(columns={'type': 'class_type'})
        label_geoms = list(map(wkt_to_shape, [superstructure for superstructure in df_labels.obstacle]))
    elif mask_generation_case == 'segments':
        df_labels = df_labels.rename(columns={'type': 'class_type'})
        class_data = [azimuth_to_label_class(az, label_classes) for az in df_labels.azimuth]
        df_labels.insert(df_labels.shape[1], 'class_type', class_data, True) #insert class_type of label to dataframe
        label_geoms = list(map(wkt_to_shape, [segment for segment in df_labels.segment]))
    elif mask_generation_case == 'pv_areas':
        df_labels = df_labels[df_labels.area.notna()]
        df_labels = df_labels.rename(columns={'type': 'class_type'})
        label_geoms = list(map(wkt_to_shape, [area for area in df_labels.area]))

    # Transform lat lon coordinates to lon lat
    label_geoms = [shapely.ops.transform(switch_coordinates, label) for label in label_geoms]

    # Add label geometry data to dataframe
    gdf_labels = gpd.GeoDataFrame(df_labels, geometry=label_geoms)
    gdf_labels.crs = 4326

    return gdf_labels


def get_gdf_images(image_folder_path, save_directory, mode='r'):
    '''
    This function generates a GeoDataframe with the image boundaries of geo-referenced images in .tif format
    In write mode, the GeoDataframe is created by opening all images and then saved to the save_directory.
    In reade mode, the function tries to read the GeoDataframe from the save_directory.
    '''

    gdf_path = save_directory + '\\data\\gdf_image_boundaries.pkl'

    if mode == 'r' and os.path.isfile(gdf_path):
        with open(gdf_path, 'rb') as f:
            gdf_images = pickle.load(f)

    elif mode == 'w' or (mode == 'r' and not os.path.isfile(gdf_path)):
        image_list = os.listdir(image_folder_path)
        # initialize geodataframe of image coordinates
        gdf_images = gpd.GeoDataFrame({'name': [], 'geometry': []})
        for image_name in image_list:
            print(image_name)
            # open image
            raster_src = gdal.Open(image_folder_path + "\\" + str(image_name), gdal.GA_ReadOnly)

            # add image bounding box from geotiff to geodataframe
            ulx, xres, xskew, uly, yskew, yres = raster_src.GetGeoTransform()  # coordinates of upper left corner and resolution
            lrx = ulx + (raster_src.RasterXSize * xres)  # coordinates of lower right corner
            lry = uly + (raster_src.RasterYSize * yres)  # coordinates of lower right corner

            image_bbox = shapely.geometry.box(ulx, lry, lrx, uly)
            gdf_images = gdf_images.append({'name': image_name, 'geometry': image_bbox}, ignore_index=True)

        gdf_images.crs = 4326

        with open(gdf_path, 'wb') as f:
            pickle.dump(gdf_images, f)
    else:
        print('mode must be either r for read or w for write')

    return gdf_images


def split_gdf_data_set(gdf_data, gdf_split_geometry):
    '''
    This function splits a GeoDataframe (gdf) into a gdf with geometries inside the split geometry and a gdf with
    elements outside the split geometry. Can be used to create a validation and a test set representation as a gdf.

    :param gdf_data:
    :param gdf_split_geometry:
    :return:

    '''

    intersections = gpd.tools.sjoin(gdf_data, gdf_split_geometry, how="inner", op='intersects')

    ids_inside = intersections.id_left.unique()
    is_outside = [id_inside == False for id_inside in gdf_data.id.isin(ids_inside)]

    split_inside = gdf_data[gdf_data.id.isin(ids_inside)]
    split_outside = gdf_data[is_outside]

    return split_outside, split_inside


def validation_split_around_location(center_coordinate, gdf_data, val_split_share):
    '''
    This function divides a dataset of geo-referenced images into training and validation set by iteratively increasing
    the validation split area around a selected center point
    '''
    # define buffer and buffer increase
    buffer_increase = 0.00025
    buffer = 0.0005
    val_split = 0
    count = 0

    # as long as validation set is too small
    while val_split < val_split_share:
        # increase buffer size around center coordinate
        buffer += buffer_increase
        count += 1
        # calculate split area
        val_selection_area = center_coordinate.buffer(buffer)
        # write split area to gdf (required for split function)
        gdf_val_selection_area = gpd.GeoDataFrame({'id': [], 'geometry': []})
        gdf_val_selection_area = gdf_val_selection_area.append({
            'id': 0, 'geometry': val_selection_area}, ignore_index=True)
        gdf_val_selection_area.crs = 4326

        # split gdf of training images into train and test
        train_set, val_set = split_gdf_data_set(gdf_data, gdf_val_selection_area)

        # calculate relative validation set size
        val_split = len(val_set)/(len(train_set)+len(val_set))

        # make sure not to loop forever
        if count >= 100:
            print('Validation set could not be created within 100 iterations')
            break

    return train_set, val_set, buffer


def relocate_segmentation_model_images_and_masks(image_id_list, save_dir, png_origin_dir, mask_origin_dir,
                                                 split_type, segmentation_classes):
    for count, image_id in enumerate(image_id_list):
        progress_string = get_progress_string(round(count/len(image_id_list), 2))
        print('Transforming vector label to mask: ' + progress_string, end="\r")
        # move png from origin to target directory
        shutil.copy(png_origin_dir + "\\" + str(image_id) + ".png",
                    save_dir + "\\" + split_type)

        # move mask of all classes from origin to target directory
        shutil.copy(mask_origin_dir + "\\"  + str(image_id) + ".png",
            save_dir + "\\" + split_type + "_masks\\")

    # document the train, val and test split in txt file
    doc_txt_file_path = save_dir + '\\' + split_type + '.txt'
    with open(doc_txt_file_path, 'w') as f:
        for image_id in image_id_list:
            f.write(str(os.path.normpath(save_dir + '\\' + split_type + '\\' + str(image_id) + '.png')) + "\n")

    return


def mask_filter(image, image_bbox, image_bbox_px, label_polygon, label_classes):
    filter_mask = np.zeros(512, 512)

    with open('data\\gdf_pvareas_annotation_experiment.pkl', 'rb') as f:
        gdf_pv_areas_AE = pickle.load(f)

    gdf_pv_areas_AE = gdf_pv_areas_AE.to_crs(4326)

    pv_areas_in_image = gdf_pv_areas_AE.intersects(image_bbox)
    pv_areas_in_image = gdf_pv_areas_AE[pv_areas_in_image]
    pv_areas_in_image = pv_areas_in_image.unary_union
    if type(pv_areas_in_image) == Polygon:
        pv_areas_in_image = MultiPolygon([pv_areas_in_image])

    for pv_area_in_image in pv_areas_in_image:
        polygon_points = convert_between_latlon_and_pixel(label_polygon, image_bbox_px,
                                                          image_bbox, case='latlon_to_px')
        polygon_points = np.int32(np.array(polygon_points))

        # Update image with label
        filter_mask = cv2.fillPoly(filter_mask, pts=[polygon_points], color=(int(1)))

    image[filter_mask == 0] = (len(label_classes))
    return image


def vector_labels_to_masks(file_vector_labels, dir_binary_masks, mask_generation_case, label_classes, gdf_images, filter=False):
    # Import labels as vector data
    gdf_labels = import_vector_labels(file_vector_labels, mask_generation_case, label_classes)

    # Make directory, if not existent
    if not os.path.isdir(dir_binary_masks):
        os.mkdir(dir_binary_masks)

    # create mask for each image
    print('\n')
    for count, gdf_image in enumerate(gdf_images.iloc):
        progress_string = get_progress_string(round(count/len(gdf_images), 2))
        print('Transforming vector label to mask: ' + progress_string, end="\r")

        image_bbox = gdf_image.geometry
        image_id = gdf_image.id
        ###############################################
        #### Match labels and images
        labels_in_image = gdf_labels.intersects(image_bbox)
        labels_in_image = [i for i, l in enumerate(labels_in_image) if l == True]
        gdf_labels_in_image = gdf_labels.iloc[labels_in_image]

        ###############################################
        #### Create binary masks
        # Initialize background image
        # set background value to "number of classes +1"
        image = np.ones([512, 512]) * (len(label_classes))
        image_bbox_px = box(0, 0, 512, 512)

        for label_id, label_class in label_classes.items():
            # For each loop, consider labels of specific class, only
            gdf_label_class = gdf_labels_in_image[gdf_labels_in_image.class_type == label_class]
            # Initialize label image
            # If image with id contains label of specific class
            if len(gdf_label_class) > 0:
                # for each label of the specific class
                for label in gdf_label_class.geometry.iloc:
                    if type(label) == MultiPolygon: # make sure, label is of type MultiPolygon
                        for label_polygon in label:
                            # Convert geo coordinates of labels to pixel coordinates
                            polygon_points = convert_between_latlon_and_pixel(label_polygon, image_bbox_px,
                                                                              image_bbox, case='latlon_to_px')
                            polygon_points = np.int32(np.array(polygon_points))

                            # Update image with label
                            image = cv2.fillPoly(image, pts=[polygon_points], color=(int(label_id)))
                            # cv2.imshow("filledPolygon", image) # activate this line for debugging

                    elif type(label) == Polygon:
                        print('Label is passed as a Polygon. This case is not covered, yet. ')
                    else:
                        print('''Label not passed as a Polygon or MultiPolygon. 
                        This case is not covered, yet. Check how that happened''')
            # If image with id does not contain label of specific class
            else:
                nothing_happens = True

        # save label mask
        filename_mask = dir_binary_masks + "\\" + str(image_id) + ".png"
        if filter:
            image = mask_filter(image, image_bbox, image_bbox_px, label_polygon, label_classes)
        cv2.imwrite(filename_mask, image ) #* 255 / (len(label_classes)+1)
    return gdf_labels

#
# def get_gdf_test_images(file_test_images_boundary,gdf_test_images):
#     # todo: do we really need this function?
#     with open(file_test_images_boundary, 'rb') as f:
#         gdf_pv_areas = pickle.load(f)
#     gdf_pv_areas = gdf_pv_areas.to_crs(4326)
#
#     gdf_pv_areas_MP = MultiPolygon([geom for geom in gdf_pv_areas.geometry.unary_union])
#     gdf_test_images = gpd.GeoDataFrame([img for img in gdf_test_images.iloc if gdf_pv_areas_MP.contains(img.geometry.centroid)])
#     annotation_experiment_image_ids = list(gdf_test_images['id'].values)
#     # exclude id 1909, because it is similar to 1439
#     annotation_experiment_image_ids.remove('1909')
#
#     return annotation_experiment_image_ids


def train_val_test_split(gdf_test_labels, gdf_images, validation_data_center_points, label_classes, dir_images_png,
                         dir_masks, dir_target_segmentation_model_data, relocate_images=False):
    # calculate, which images overlap with annotation experiment labels --> exclude from training dataset
    gdf_train_val_images, gdf_test_images = split_gdf_data_set(gdf_images, gdf_test_labels)

    # variate center coordinate to create different train-val data set splits
    for i, center_coordinate in enumerate(validation_data_center_points):
        print('Processing validation split: ' + str(i))
        # split training set into training and validation set
        gdf_train_images, gdf_val_images, buffer = validation_split_around_location(center_coordinate, gdf_train_val_images, 0.2)

        image_id_lists = list([list(gdf_train_images.id), list(gdf_val_images.id), list(gdf_test_images.id)])
        data_set_types = ['train', 'val', 'test']

        # if images and masks should be split and copied to subdirectories train, val and test:
        if relocate_images == True:
            # create folder structure, if not existing
            dir_segmentation_model_data = dir_target_segmentation_model_data + '_' + str(i + 1)
            if not os.path.isdir(dir_segmentation_model_data):
                os.mkdir(dir_segmentation_model_data)
                sub_dirs = ['train', 'train_masks', 'val', 'val_masks', 'test', 'test_masks']
                for sub_dir in sub_dirs:
                    os.mkdir(dir_segmentation_model_data + '\\' + sub_dir)

            # relocate images and masks to train, val and test subfolders
            for j, data_set_type in enumerate(data_set_types):
                relocate_segmentation_model_images_and_masks(image_id_lists[j],
                                                             dir_segmentation_model_data,
                                                             dir_images_png,
                                                             dir_masks,
                                                             data_set_type,
                                                             label_classes)
        # else: all images and masks are in one folder. train, val and test are loaded using filenames from .txt file
        else:
            # save .txt files with respect to train, val, test split
            for j, data_set_type in enumerate(data_set_types):
                filenames = [str(id) + '.png' for id in image_id_lists[j]]
                filepath = os.path.join(dir_target_segmentation_model_data,
                                        (data_set_type + '_filenames_' + str(i + 1) + '_6_classes.txt'))
                with open(filepath, 'w') as f:
                    [f.write(filename + '\n') for filename in filenames]
    return

