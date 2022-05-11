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
import os
import pandas as pd
import geopandas as gpd
import pickle
import shapely

from definitions import LABEL_CLASSES_SUPERSTRUCTURES, LABEL_CLASSES_SEGMENTS, IMAGE_SHAPE, \
    FILE_VECTOR_LABELS_SEGMENTS, FILE_VECTOR_LABELS_SUPERSTRUCTURES, DIR_IMAGES_PNG, \
    df_technical_potential_LUT, EPSG_METRIC, DIR_PREDICTIONS
from utils import prediction_raster_to_vector, get_progress_string
from mask_generation import import_vector_labels
from module_placement import module_placement_options
from visualization import visualize_module_placement, box_plot_E_gen_TUM_CI



def technical_potential_kWp_lookup(latitude, longitude, azimuth, slope):
    '''
        This function returns the energy generation from a lookup table with data from PVGIS. The lookup table is
        generated as a batch job (see batch jobs). If there is no suitable entry in the lookup, the PCGIS API is called.

        :param latitude: float
            latitude of roof segment's location
        :param lonigtude: float
            lonigtude of roof segment's location
        :param azimuth: float
            azimuth of roof segment, between -180 (N) and 180 (N), with 0 being south. see PVGIS for definition
        :param slope: float
            roof tilt/slope angle between 0 and 90

        :return: PV_E_gen_hourly pandas Dataframe
            pandas Dataframe with time and power as colomns. Contains the hourly values of one year.
            power is in W
        '''

    df_red = df_technical_potential_LUT.loc[(((df_technical_potential_LUT.latitude - latitude) <= 0.1) &
                                             ((df_technical_potential_LUT.longitude - longitude) <= 0.1) &
                                             (df_technical_potential_LUT.azimuth == azimuth) &
                                             (df_technical_potential_LUT.slope == slope))]

    # check if lookup value was found and extract PV_E_gen_hourly, if not found: call PVGIS API
    if len(df_red) == 0:
        print('no lookup value found for given inputs: slope is ' + str(slope) + ' azimuth is ' + str(azimuth))
        PV_E_gen_yearly = np.nan
    else:
        PV_E_gen_yearly = df_red.E_gen_yearly.values[0]

    return PV_E_gen_yearly


def substract_superstructures_from_segments(gdf_segments, gdf_superstructures):
    area_list = []
    contains_pvmodule_list = []
    count = 1
    intersections = gpd.tools.sjoin(gdf_segments, gdf_superstructures, how="inner", op='intersects')
    id_list = intersections.id_left.unique()
    for id in id_list:
        segment = gdf_segments.geometry[gdf_segments.id == id]
        ids_inside = intersections[intersections.id_left==id].id_right
        # GeoDataFrame of superstructures in segment
        superstructures_in_segment = gdf_superstructures[
            [id_inside == True for id_inside in gdf_superstructures.id.isin(ids_inside)]
        ]

        contains_pvmodule_list.append((superstructures_in_segment.label_type == 'pvmodule').any())

        # unary union of superstructures in segment
        uu_superstructures = shapely.ops.unary_union(superstructures_in_segment.geometry)

        # check for validity, because unary_union can lead to self intersection polygons
        if not uu_superstructures.is_valid:
            uu_superstructures = uu_superstructures.buffer(0)
        # update area value in gdf_segments as segment area minus area of superstructure in segments
        area_list.append(segment.difference(uu_superstructures).area)

        # show processing progress
        count += 1
        progress_string = get_progress_string(round(count / len(id_list), 2))
        print('processing segment #' + str(count) + ' ' + progress_string, end='\r')

    return area_list, contains_pvmodule_list, id_list


# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------ #
# 1) get roof superstructures
# ------------------------------------------------------------------------------------------------------------------ #
def pv_potential_analysis():
    # a) load superstructure predictions
    prediction_mask_filenames = os.listdir(DIR_PREDICTIONS)
    prediction_mask_filepaths = [os.path.join(DIR_PREDICTIONS, file) for file in prediction_mask_filenames]
    prediction_masks = [cv2.imread(prediction, 0) for prediction in prediction_mask_filepaths]

    # b) load superstructure ground truth
    gdf_superstructures_GT = import_vector_labels(
        FILE_VECTOR_LABELS_SUPERSTRUCTURES,
        'superstructures',
        LABEL_CLASSES_SUPERSTRUCTURES
    )
    # rename class colomn
    gdf_superstructures_GT = gdf_superstructures_GT.rename(columns={'class_type': 'label_type'})

    # transform coordinate system
    gdf_superstructures_GT.crs = 4327
    gdf_superstructures_GT = gdf_superstructures_GT.to_crs(EPSG_METRIC)

    with open("data\\gdf_image_boundaries.pkl", 'rb') as f:
        gdf_images = pickle.load(f)
    gdf_images.id = gdf_images.id.astype(int)

    gdf_superstructures_PR = gpd.GeoDataFrame({'mask_id': [], 'label_type': [], 'geometry': []})

    # # c) convert raster data of superstructures to vector data
    for i, mask in enumerate(prediction_masks):
        mask_id = prediction_mask_filenames[i][:-4]
        gdf_image = gdf_images[gdf_images.id == int(mask_id)]
        image_bbox = gdf_image.geometry.iloc[0]
        gdf_predictions = prediction_raster_to_vector(mask, mask_id, image_bbox, LABEL_CLASSES_SUPERSTRUCTURES, IMAGE_SHAPE)
        # add new prediction labels to superstructure dataframe
        gdf_superstructures_PR = pd.concat([gdf_superstructures_PR, gdf_predictions], axis=0)

    gdf_superstructures_PR.crs = 4327
    gdf_superstructures_PR = gdf_superstructures_PR.to_crs(EPSG_METRIC)
    # add an id colomn, important for matching to segment later
    gdf_superstructures_PR.insert(0, 'id', np.arange(0, len(gdf_superstructures_PR)))

    # ------------------------------------------------------------------------------------------------------------------ #
    # 2) get roof segments
    # ------------------------------------------------------------------------------------------------------------------ #
    gdf_segments_GT = import_vector_labels(FILE_VECTOR_LABELS_SEGMENTS, 'segments', LABEL_CLASSES_SEGMENTS)
    gdf_segments_GT.crs = 4327 # set coordinate system
    # convert to metric coordinate system
    gdf_segments_GT = gdf_segments_GT.to_crs(EPSG_METRIC)

    # drop unnecessary columns
    gdf_segments_GT.drop(gdf_segments_GT.columns[[0, 1, 4, 5, 6]], axis=1, inplace=True)

    # drop all segments which are not part of case study (prediction masks)
    mask_ids = [mask_id[:-4] for mask_id in prediction_mask_filenames]
    gdf_segments_GT = gdf_segments_GT[gdf_segments_GT.pv_area_id.isin(mask_ids)]

    # ------------------------------------------------------------------------------------------------------------------ #
    # 3) determine PV related roof area (simple area w/ or w/o superstructure, module placement)
    # ------------------------------------------------------------------------------------------------------------------ #
    gdf_segments_GT.slope[gdf_segments_GT.slope != 0] = 30
    gdf_segments_GT.insert(4, 'is_flat', gdf_segments_GT.slope == 0)

    # # a) simple area w/o superstructures
    area_list = np.round(np.array(gdf_segments_GT.geometry.area), 2)
    A_1 = area_list.copy()

    # b) simple area w/ superstructures (GT)
    area_list, contains_pvmodule_list, id_list \
        = substract_superstructures_from_segments(gdf_segments_GT, gdf_superstructures_GT)
    area_list = [area.values[0] for area in area_list]

    A_2 = A_1.copy()
    iloc_list = [list(gdf_segments_GT.id).index(id) for id in id_list]
    A_2[iloc_list] = np.round(np.array(area_list), 2)

    # c) simple area w/ superstructures (PR)
    area_list, contains_pvmodule_list, id_list = \
        substract_superstructures_from_segments(gdf_segments_GT, gdf_superstructures_PR)
    area_list = [area.values[0] for area in area_list]

    A_3 = A_1.copy()
    iloc_list = [list(gdf_segments_GT.id).index(id) for id in id_list]
    A_3[iloc_list] = np.round(np.array(area_list), 2)

    # d) simple area w/ superstructures (PR) & substraction of existing PV (PR)
    helper = [id == False for id in contains_pvmodule_list]
    area_list = np.array(area_list) * helper

    A_4 = A_1.copy()
    iloc_list = [list(gdf_segments_GT.id).index(id) for id in id_list]
    A_4[iloc_list] = np.round(np.array(area_list), 2)

    # e) module placement without superstructures
    _ , _, num_modules_list_no_superstructures, \
    modules_shape_list_no_superstructures, _ = \
        module_placement_options(
            gdf_segments_GT,
            gdf_superstructures_GT[0:1],
            1.6, 1, [], []
    )

    A_5 = [modules.area for modules in modules_shape_list_no_superstructures]

    # f) module placement with superstructures
    gdf_segments_GT , _, num_modules_list, \
    modules_shape_list, _ = \
        module_placement_options(
            gdf_segments_GT,
            gdf_superstructures_PR,
            1.6, 1, [], []
    )

    # write module shapes to GeoDataFrame
    gdf_modules_PR = gpd.GeoDataFrame({'num': num_modules_list, 'geometry' : modules_shape_list})
    gdf_modules_PR.crs = EPSG_METRIC

    A_6 = [modules.area for modules in modules_shape_list]

    # adapt areas according to slope: projection of tilted area is smaller than real area
    area_projection_correction = np.array([np.cos(slope / 360 * 2 * np.pi) for slope in gdf_segments_GT.slope])
    A_list = [A_1, A_2, A_3, A_4]
    # adapt all segment areas according to slope value (flat roofs are not projected into the tilted plane)
    for i, A in enumerate(A_list):
        gdf_segments_GT['A_' + str(i+1)] = np.array(A) / area_projection_correction

    # transform modules back from projection to tilted plane: all modules have a slope of 30°, on flat and tilted roofs
    A_list = [A_5, A_6]
    for i, A in enumerate(A_list):
        gdf_segments_GT['A_' + str(i+5)] = np.array(A) / np.cos(30 / 360 * 2 * np.pi)

    with open('results\\res_pv_potential.pkl', 'wb') as f:
        pickle.dump([gdf_segments_GT, gdf_superstructures_PR, gdf_superstructures_GT, gdf_modules_PR, modules_shape_list,
                     modules_shape_list_no_superstructures], f)

    # ------------------------------------------------------------------------------------------------------------------ #
    # 4) calculate technical potential
    # round azimuth values to multiples of 3 (for technical potential lookup table)
    gdf_segments_GT.azimuth = [np.round(az/3, 0) * 3 for az in gdf_segments_GT.azimuth.iloc]
    gdf_segments_GT.azimuth[gdf_segments_GT.azimuth == 180] = -180
    gdf_segments_GT.azimuth[gdf_segments_GT.azimuth.isna()] = 0

    # calculate potential energy production per segment:
    E_gen_yearly_per_kWp = [technical_potential_kWp_lookup(48.401, 11.978, seg.azimuth, seg.slope)
                            for seg in gdf_segments_GT.iloc]

    gdf_segments_GT['E_kWp'] = E_gen_yearly_per_kWp

    # scale energy generation per year according to area: 1kWp per 4 m²,
    for i in [1, 2, 3, 4, 5, 6]:
        gdf_segments_GT['E_'+str(i)] = gdf_segments_GT['A_'+str(i)] / 4 * gdf_segments_GT['E_kWp']
    # for flat roofs 1 kWp per 8 m²
    for i in [1, 2, 3, 4]:
        gdf_segments_GT['E_'+str(i)][gdf_segments_GT.is_flat==True] = gdf_segments_GT['A_'+str(i)] / 8 * \
                                                                      gdf_segments_GT['E_kWp']

    # visualize
    image_ids = [533, 257, 355]
    visualize_module_placement(image_ids, gdf_images, gdf_segments_GT, gdf_superstructures_PR, gdf_modules_PR,
                               DIR_IMAGES_PNG)

    box_plot_cols = ['E_1', 'E_2', 'E_3', 'E_4', 'E_5', 'E_6']
    box_plot_x_ticks = ['without superstr.', 'with superstr. \n (GT)', 'with superstr. \n (PR)',
                        'with superstr. \n (PR) \n excl. existing solar', 'PV modules \n without \n superstr.',
                        'PV modules \n with \n superstr. (PR)']

    E_specific = [gdf_segments_GT[col] / gdf_segments_GT['A_1'] for col in box_plot_cols]

    E_sums = np.round([np.sum(gdf_segments_GT[col]) / 1000 / 1000 for col in box_plot_cols], 2)
    box_plot_E_gen_TUM_CI(E_specific, box_plot_x_ticks, E_sums, 'Segment energy generation \n in kWh / (a m²) ',
                    'plots\\pv_potential.svg', colors=[], image_size='one_third_large', show_means=True)

    return

