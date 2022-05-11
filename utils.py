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
import overpy
import pickle
import shapely

import geopandas as gpd
import numpy as np
import pandas as pd
from osgeo import gdal
from osgeo import osr
from shapely.geometry import Point, LineString, Polygon, MultiLineString, MultiPolygon, box



def geotif_to_png(dir_geotif, dir_png):
    '''Compares directories and converts geotif to png, if the respective image does not exist in png directory'''
    # make sure png path exists
    if not os.path.isdir(dir_png):
        os.mkdir(dir_png)
    # compare image file list
    files_geotif = [geotif[:-4] for geotif in os.listdir(dir_geotif) if geotif[-4:] == '.tif']
    files_png = [png[:-4] for png in os.listdir(dir_png) if png[-4:] == '.png']

    missing_pngs_list = [geotif for geotif in files_geotif if geotif not in files_png]

    # open geotif and save as png
    for i, img in enumerate(missing_pngs_list):
        progress_string = get_progress_string(round(i / len(missing_pngs_list), 2))
        print('Converting geotif to pngs ' + progress_string, end="\r")

        png_file_path = os.path.join(dir_png, img + '.png')
        geotif_file_path = os.path.join(dir_geotif, img + '.tif')

        image = cv2.imread(geotif_file_path)
        cv2.imwrite(png_file_path, image)
    return


def azimuth_to_label_class(az, label_classes):
    label_classes = label_classes[:-1]
    if np.isnan(az):
        az_class = "flat"
    else:
        surplus_angle = 360 / len(label_classes) / 2
        az = az + 180 + surplus_angle
        if az > 360: az -= 360
        az_id = int(np.ceil(az / (360 / len(label_classes))) - 1)
        az_class = label_classes[az_id]
    return az_class


def get_progress_string(progress):
    number_of_dashes = 100
    progress_string = '| >'
    progress_string += int(round(progress * number_of_dashes, 0)) * '>' + \
                       int(round((1 - progress) * number_of_dashes, 0)) * '-'
    progress_string += ' |'
    return progress_string


def wkt_to_shape(wkt_str):
    ''' function to transform a shape from WKT format to shape format'''
    return shapely.wkt.loads(wkt_str)


def switch_coordinates(x, y):
    ''' function to switch x and y coordinates'''
    return y, x


def convert_lonlat_to_latlon(obj):
    ''' function to switch longitude, latitude coordinates to latitude, longitude coordinates'''
    return shapely.ops.transform(switch_coordinates, obj)


def convert_latlon_to_lonlat(obj):
    ''' function to switch latitude, longitude coordinates to longitude, latitude coordinates'''
    return shapely.ops.transform(switch_coordinates, obj)


def convert_lonlat_to_metric(obj):
    WGS84_EPSG = {'init': 'epsg:4326'}  # long lat: coordinates need to be in Long Lat, instead of Lat Long
    # METRIC_EPSG = {'init': 'epsg:3857'}  #
    METRIC_EPSG = {'init': 'epsg:25832'} # UTM 32
    # METRIC_EPSG = {'init': 'epsg:31468'}  # Gauss-Kruger Zone 4
    obj = gpd.GeoSeries(obj)
    obj.crs = WGS84_EPSG
    obj = obj.to_crs(METRIC_EPSG)
    obj = obj.geometry
    return obj


def convert_metric_to_lonlat(obj):
    WGS84_EPSG = {'init': 'epsg:4326'}  # lat lon
    # METRIC_EPSG = {'init': 'epsg:3857'}  #
    METRIC_EPSG = {'init': 'epsg:25832'} # UTM 32
    # METRIC_EPSG = {'init': 'epsg:31468'}  # Gauss-Kruger Zone 4

    obj = gpd.GeoSeries(obj)
    obj.crs = METRIC_EPSG
    obj = obj.to_crs(WGS84_EPSG)
    obj = obj.geometry
    return obj


def convert_between_latlon_and_pixel(geom, image_px, image_latlon, case='latlon_to_px'):
    # todo: describe function
    if case == 'px_to_latlon':
        image_orig = image_px
        image_target = image_latlon
    elif case == 'latlon_to_px':
        image_orig = image_latlon
        image_target = image_px
    else:
        print('transformation case not covered: try case=px_to_latlon ')

    iox_min = image_orig.bounds[0]
    iox_max = image_orig.bounds[2]
    itx_min = image_target.bounds[0]
    itx_max = image_target.bounds[2]

    ioy_min = image_orig.bounds[1]
    ioy_max = image_orig.bounds[3]
    ity_min = image_target.bounds[1]
    ity_max = image_target.bounds[3]

    if type(geom) == MultiPolygon or type(geom) == MultiLineString:
        print('Geometry type not allowed')
    elif type(geom) == Polygon:
        coords = geom.exterior.coords
    elif type(geom) == LineString:
        coords = geom.coords

    point_list = []
    for x, y in coords:
        x_new = itx_min + ((x - iox_min) / (iox_max - iox_min) * (itx_max - itx_min))
        y_new = ity_min + ((ioy_max - y) / (ioy_max - ioy_min) * (ity_max - ity_min))
        point_list.append((x_new, y_new))
    return point_list


def save_as_geotif(bbox, image, save_path):
    """
    Function to save a jpg or png as GEOTIFF

    Inputs
    ----------
    bbox : list
        bounding box coordinates of image:
        [latitude_min, longitude_min, latitude_max, longitude_max]

    image : numpy array
        RGB image as a numpy array with shape [x_pixels, y_pixels, 3]

    save_path : string
        string that determines the path including filename to save the GEOTIFF
        filename should end with ".tif"

    Outputs
    ----------
    None - function saves GEOTIFF to path
    """
    # set geotransform
    nx = image.shape[0]
    ny = image.shape[1]

    xmin, ymin, xmax, ymax = [bbox[1], bbox[0], bbox[3], bbox[2]]
    xres = (xmax - xmin) / float(nx)
    yres = (ymax - ymin) / float(ny)

    geotransform = (xmin, xres, 0, ymax, 0, -yres)

    # create the 3-band raster file
    dst_ds = gdal.GetDriverByName('GTiff').Create(save_path, ny, nx, 3, gdal.GDT_Byte)

    dst_ds.SetGeoTransform(geotransform)  # specify coords
    srs = osr.SpatialReference()  # establish encoding
    srs.ImportFromEPSG(4326)  #
    dst_ds.SetProjection(srs.ExportToWkt())  # export coords to file
    dst_ds.GetRasterBand(1).WriteArray(image[:, :, 2])  # write r-band to the raster
    dst_ds.GetRasterBand(2).WriteArray(image[:, :, 1])  # write g-band to the raster
    dst_ds.GetRasterBand(3).WriteArray(image[:, :, 0])  # write b-band to the raster
    dst_ds.FlushCache()  # write to disk

    return



def get_static_map_bounds(latitude, longitude, zoom, width, height):
    """ Get the bounds for the image from the image centerpoint.
    Only works for high zoom levels because of simplifications.

    Parameters
    ----------
    latitude : numeric
        Latitude value of centerpoint.

    longitude : nuemric
        Longitude value of centerpoint.

    zoom : numeric
        Zoom level.

    width : numeric
        Image width.

    height : numeric
        Image height

    Returns
    ----------
    bbox : tuple
        (Min_latitude, min_longitude, max_latitude, max_longitude)
    """

    # 256 pixels - initial map size for zoom factor 0
    size = 256 * 2 ** zoom

    # resolution in degrees per pixel
    res_lat = np.cos(latitude * np.pi / 180.) * 360. / size
    res_lng = 360. / size

    d_lat = res_lat * height / 2
    d_lng = res_lng * width / 2

    bbox = (
        latitude - d_lat,
        longitude - d_lng,
        latitude + d_lat,
        longitude + d_lng)
    return bbox


def get_image_gdf_in_directory(DIR_IMAGES_GEOTIFF, save_to_png_path=[]):
    image_id_list = [id[:-4] for id in os.listdir(DIR_IMAGES_GEOTIFF) if id[-4:] == '.tif']

    # open image
    raster_srcs = [gdal.Open(DIR_IMAGES_GEOTIFF + "\\" + str(image_id) + ".tif", gdal.GA_ReadOnly) for image_id in
            image_id_list]
    image_bbox_list = []
    print('')
    for i, image_id in enumerate(image_id_list):
        raster_src = raster_srcs[i]
        progress_string = get_progress_string(round(i/len(raster_srcs), 2))
        print('Loading geo_tiffs: ' + progress_string, end="\r")
        # print('Currently processing image # ' + str(count) + '/' + str(len(raster_srcs)), end="\r")

        # initialize rgb image with shape of .tif
        if len(save_to_png_path) > 0:
            filename_mask = save_to_png_path + str(image_id) + '.png'
            data = raster_src.ReadAsArray()
            img = np.dstack((data[0, :, :], data[1, :, :], data[2, :, :]))
            cv2.imwrite(filename_mask, img)
        #
        # band_shape = np.shape(raster_src.GetRasterBand(1).ReadAsArray())
        # image_rgb = np.zeros([band_shape[0], band_shape[1], 3])
        #
        # # write .tif bands to rgb
        # for i in np.arange(0, 3):
        #     image_rgb[:, :, i] = raster_src.GetRasterBand(int(i + 1)).ReadAsArray() / 256

         # add image bounding box from geotiff to geodataframe
        ulx, xres, xskew, uly, yskew, yres = raster_src.GetGeoTransform()  # coordinates of upper left corner and resolution
        lrx = ulx + (raster_src.RasterXSize * xres)  # coordinates of lower right corner
        lry = uly + (raster_src.RasterYSize * yres)  # coordinates of lower right corner

        image_bbox = shapely.geometry.box(ulx, lry, lrx, uly)
        image_bbox_list.append(image_bbox)
    image_id_list = [int(id) for id in image_id_list]
    # initialize geodataframe of image coordinates
    gdf_images = gpd.GeoDataFrame({'id': image_id_list, 'geometry': image_bbox_list})
    # todo: save all gdf with their crs already
    gdf_images.crs = 4326
    return gdf_images


def get_wartenberg_boundary():
    file_path_boundary = 'data\\gdf_Wartenberg_boundary.pkl'
    # if file of boundary does not exist: get boundary from osm and save
    if not os.path.isfile(file_path_boundary):
        # get way points from osm
        r = overpy.Overpass().query('[out:json]; relation(934742); (._;>;); out;')
        ls = []
        for w in r.ways:
            ls.append(LineString([Point(float(node.lon), float(node.lat)) for node in w.nodes]))
        # polygonize seperate ways into one multipolygon
        pol1 = shapely.ops.polygonize_full(ls[0])[0][0]  # (first way is a polygon already, other ways are linestrings)
        pol2 = shapely.ops.polygonize_full(ls[1:])[0][0]
        mp_Wartenberg = MultiPolygon([pol1, pol2])
        # write to geodataframe and save
        gdf_Wartenberg_boundary = gpd.GeoDataFrame({'name': 'Wartenberg boundary', 'geometry': [mp_Wartenberg]})
        with open(file_path_boundary, 'wb') as f:
            pickle.dump(gdf_Wartenberg_boundary, f)
    # otherwhise, just load
    else:
        with open(file_path_boundary, 'rb') as f:
            gdf_Wartenberg_boundary = pickle.load(f)

    return gdf_Wartenberg_boundary


def metrics_from_confusion_matrix(CM):
    FP = CM.sum(axis=0) - np.diag(CM)
    FN = CM.sum(axis=1) - np.diag(CM)
    TP = np.diag(CM)
    TN = CM.sum() - (FP + FN + TP)
    IoU = TP/(TP+FP+FN)
    mean_IoU = np.nanmean(IoU)
    Accuracy = (TP+TN)/np.nansum([TP, TN, FP, FN])
    Precision = TP/np.nansum([TP, FP])
    Recall = TP/np.nansum([TP, FN])
    F1 = (2*TP)/np.nansum([2*TP, FN, FP])
    return IoU, mean_IoU, Accuracy, Precision, Recall, F1


def normalize_confusion_matrix_by_rows(CM):
    row_sum = np.sum(CM, axis=1)
    CM_normalized = np.zeros(CM.shape, dtype=np.float)
    for i, row in enumerate(row_sum):
        if row > 0:
            CM_normalized[i, :] = CM[i, :] / row
    return CM_normalized


def df_IoU_from_confusion_matrix(CM_list, CLASSES):
    METRICS = [metrics_from_confusion_matrix(cm) for cm in CM_list]

    mean_IoUs = [m_iou[1] for m_iou in METRICS]
    class_IoUs = [iou[0] for iou in METRICS]

    df_IoUs = pd.DataFrame()
    for cl_id, cl_name in enumerate(CLASSES):
        ious = [cl_iou[cl_id] for cl_iou in class_IoUs]
        df_IoUs[cl_name] = ious
    df_IoUs = df_IoUs.assign(mean_IoU=mean_IoUs)

    return df_IoUs


def prediction_raster_to_vector(prediction_mask, mask_id, image_bbox, CLASSES, IMAGE_SHAPE):
    """
    Takes the roof image as input and outputs the supersturctures polygon
    Parameters
    ----------
    image : nd-array
        prediction_mask .

    Returns
    ----------
    polygons : list
        List of Shapely polygons
    """

    label_type_list = []
    geometry_list = []

    for i, class_id in enumerate(CLASSES):
        prediction = np.zeros(IMAGE_SHAPE)
        prediction[prediction_mask==class_id] = 1
        prediction = prediction.astype(np.uint8)

        if np.sum(prediction) > 0:
            # get the contours from the mask
            contours, _ = cv2.findContours(
                prediction, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Convert the contours list to a Numpy array
            contours = np.array(contours, dtype=object)

            # convert the contours to shapely polygon
            for cnt in contours:
                cnt = cnt.reshape(-1, 2)
                try:
                    shapely_poly = Polygon(cnt)
                except ValueError:
                    continue
                # write the results to lists
                geometry_list.append(shapely_poly)
                label_type_list.append(CLASSES[i])

    image_bbox_px = box(0, 0, IMAGE_SHAPE[0], IMAGE_SHAPE[1])
    geometry_list = [convert_between_latlon_and_pixel(geometry, image_bbox_px, image_bbox, case='px_to_latlon')
                     for geometry in geometry_list]
    geometry_list = [Polygon(geometry) for geometry in geometry_list]

    gdf_labels = gpd.GeoDataFrame({
        'mask_id': list([mask_id]) * len(label_type_list),
        'label_type': label_type_list,
        'geometry': geometry_list
    })

    return gdf_labels