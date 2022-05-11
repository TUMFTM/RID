__author__ = "Sebastian Krapf"
__copyright__ = "Copyright 2021, "
__credits__ = []
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Sebastian Krapf"
__email__ = "sebastian.krapf@tum.de"
__status__ = "alpha"

from osgeo import gdal, osr
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon, Point, LineString
from utils import get_wartenberg_boundary, get_static_map_bounds, save_as_geotif
import geopandas as gpd
import cv2

import shapely
shapely.speedups.disable()


from definitions import \
    FILE_VECTOR_LABELS_SUPERSTRUCTURES,\
    FILE_VECTOR_LABELS_SEGMENTS,\
    FILE_VECTOR_LABELS_PV_AREAS,\
    DIR_BASE

# todo: clean up file path chaos
# input_file_path = "C:\\00_All\\40_Python\\Masks_CNN_roofs\\testimage.png"
# file_folder_path = "W:\\Projekte\\Firefly\\Johanna_Prummer\\Masken\\210822_Segments\\Images\\Imgs_Grid17_corrected"
file_folder_path = "W:\\Projekte\\Firefly\\Team-Rooftop PV\\02_Data\\Base_Images_Wartenberg_per_ID"
# file_folder_path = "C:\\Users\\ga73pag\\Desktop\\wartenberg\\output\\partial"
image_paths = os.listdir(file_folder_path)
# save_geotiff_directory = "C:\\00_All\\40_Python\\Masks_CNN_roofs\\images_grid_geotiff"
# save_geotiff_directory = "C:\\00_All\\40_Python\\Masks_CNN_roofs\\images_grid_repaired_geotiff_partial"
save_geotiff_directory = DIR_BASE + "\\images_roof_centered_geotiff"
# csv_image_coordinates_path = "C:\\00_All\\40_Python\\Masks_CNN_roofs\\centroids_Grid17.csv"
csv_image_coordinates_path = DIR_BASE + "\\data\\centroids_centered_images.csv"

geotiff_source_directory = "C:\\img_gtiff"

# use if image coordinates are saved in csv file
# image_coordinates = pd.read_csv(csv_image_coordinates_path)

# if image coordinates are included in filename:
# image_longitude_list = [float(coords[0]) for coords in [image.split('_') for image in image_paths]]
# image_latitude_list = [coords[1] for coords in [image.split('_') for image in image_paths]]
# image_latitude_list = [float(lat.replace('.png', '')) for lat in image_latitude_list]
#
# image_coordinates = pd.DataFrame({'longitude': image_longitude_list, 'latitude': image_latitude_list})
#### Images from Google Earth ###
ge_img_filepath = "C:\\Users\\ga73pag\\Desktop\\georef\\01.tif"
def get_image_bound_google_earth(img_center):
    '''
    get image boundary for image from google earth
    - image resolution must be  4800 x 4800 px
    - camera height 406 m

    Inputs
    ----------
    img_center : tuple
        center point of image in EPSG 4326

    Outputs
    ----------
    bbox : list
        list of four coordinates in EPSG 4326:
        [lon_min, lat_min, lon_max, lat_max]
    '''
    ## for 1028 x 1028 px
    # delta_x_1028 = 0.0013581095823997913
    # delta_y_1028 = 0.0009032451282351417

    delta_x = 0.0063444141746469285
    delta_y = 0.0042121400787706875

    bbox = [img_center[0] - 0.5 * delta_x,
            img_center[1] - 0.5 * delta_y,
            img_center[0] + 0.5 * delta_x,
            img_center[1] + 0.5 * delta_y]
    return bbox

### get all center points
# load example image
raster_src = gdal.Open(ge_img_filepath, gdal.GA_ReadOnly)

# add image bounding box from geotiff to geodataframe
ulx, xres, xskew, uly, yskew, yres = raster_src.GetGeoTransform()  # coordinates of upper left corner and resolution
lrx = ulx + (raster_src.RasterXSize * xres)  # coordinates of lower right corner
lry = uly + (raster_src.RasterYSize * yres)  # coordinates of lower right corner
image_bbox = shapely.geometry.box(ulx, lry, lrx, uly)
gdf_img = gpd.GeoDataFrame({'id': [1], 'geometry': [image_bbox]})
gdf_img.crs = 3857
gdf_img = gdf_img.to_crs(4326)

# calculate delta x and delta y for an image of 4800 x 4800 px
height = 87
ge_center = gdf_img.geometry.centroid.iloc[0]
coords = gdf_img.iloc[0].geometry.exterior.xy
delta_lon = max(coords[0]) - min(coords[0])
delta_lat = max(coords[1]) - min(coords[1])

# calulcate center points of all images
latitude_start = 48.395063
longitude_start = 11.975436
latitude_end = 48.414266
longitude_end = 12.003409

lat = latitude_start
lon = longitude_start

gdf_center_points = gpd.GeoDataFrame({'id': [], 'lat:': [], 'lon': [], 'geometry': [], 'height': []})
center_point_list = []
count = 1
while lat <= latitude_end + delta_lat:
    lon = longitude_start
    while lon <= longitude_end + delta_lon:
        add_point = Point(lat, lon)
        center_point_list.append(add_point)
        gdf_center_points = gdf_center_points.append({'id': count,
                                                      'lat:': lat,
                                                      'lon': lon,
                                                      'geometry': add_point,
                                                      'height': height},
                                                     ignore_index=True)
        lon += delta_lon
        count += 1
    lat += delta_lat

# make a csv file from points
gdf_center_points.to_csv('center_points_google_earth.csv')

# save image as geotiff
dir_images = "C:\\Users\\ga73pag\\Desktop\\georef\\"
dir_tiffs = dir_images + 'tif'
dir_jpgs = dir_images + 'png'
files_jpgs = os.listdir(dir_jpgs)
for i, file_png in enumerate(files_jpgs):
    center_point = center_point_list[i]
    bbox = get_image_bound_google_earth((center_point.y, center_point.x))
    bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]

    image = cv2.imread(dir_jpgs + '\\' + str(i + 1) + '.jpg')
    image = cv2.resize(image, (4608, 4608), interpolation=cv2.INTER_AREA)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    file_tif = dir_tiffs + '\\' + str(i+1) + '.tif'

    save_as_geotif(bbox, image, file_tif)


######################## split large tifs #############################################
file_list = os.listdir(dir_tiffs)
for file_tif in file_list:
    cmd = "gdal_retile.py -ps 512 512 -targetDir " + dir_tiffs[:-3] + "small_tifs" + " " + dir_tiffs + "\\" + file_tif
    print(os.popen(cmd).read())


# Define area of interest
gdf_Wartenberg_boundary = get_wartenberg_boundary()

######################## filter test labels #############################################
df_pv_areas = pd.read_csv(DIR_BASE + "\\" + FILE_VECTOR_LABELS_PV_AREAS)
df_pv_segments = pd.read_csv(DIR_BASE + "\\" + FILE_VECTOR_LABELS_SEGMENTS)
df_pv_superstructures = pd.read_csv(DIR_BASE + "\\" + FILE_VECTOR_LABELS_SUPERSTRUCTURES)

df_pv_segments = df_pv_segments[df_pv_segments['pv_area_id'].isin(df_pv_areas.id)]
df_pv_superstructures = df_pv_superstructures[df_pv_superstructures['pv_area_id'].isin(df_pv_areas.id)]

# df_pv_segments.to_csv('segments.csv')
# df_pv_superstructures.to_csv("obstacles.csv")

######################## Merge geotiffs #############################################
merge = False
if merge == True:
    geo_tiff_paths = os.listdir(geotiff_source_directory)

    files_string = ""
    count = 0
    num_grouped = 1835
    for i, geo_tiff_path in enumerate(geo_tiff_paths):
        # count += 1
        # if count < num_grouped:
        files_string = files_string + "" + geotiff_source_directory + "\\" + geo_tiff_path + "\n"
        # else:
        # count = 1
        # group_count = int(i/num_grouped)
        # files_string = files_string + " " + geotiff_source_directory + "\\" + geo_tiff_path
    # input_geotiffs.txt
    command_vrt = "gdalbuildvrt -input_file_list input_geotiffs.txt VRT_WB.vrt"
    command = "gdal_merge.py -o merged_tiff.tif -of gtiff VRT_WB.vrt"
    print(len(command))
    print(os.popen(command_vrt).read())
    files_string = ""

#gdal_retile.py -ps 512 512 -targetDir C:\example\dir *.tif

# for i, image_path in enumerate(image_paths):
    # use this, if there is an image id given in csv file
for image_coordinate in image_coordinates.iloc:
    # print(image_coordinate.id)

    #  Choose some Geographic Transform
    lat = image_coordinate.latitude
    lon = image_coordinate.longitude

    # filter out labels outside the area of interest
    if not gdf_Wartenberg_boundary.geometry.contains(Point(lon, lat)).any():
        print(str(image_coordinate.id) + 'is not within Wartenberg boundary')
    else:
        image_path = str(int(image_coordinate.id)) + '.png'

        # image_coordinate = image_coordinates.iloc[i]
        # image_filename = str("%.6f" %image_coordinate.longitude) + '_' + str("%.6f" %image_coordinate.latitude)
        # print(image_filename)

        image_path = file_folder_path + "\\" + image_path
        image_filename = str(int(image_coordinate.id))

        if os.path.isfile(image_path):
            image = plt.imread(image_path)
            image = np.round(image*255)

            # set geotransform
            nx = image.shape[0]
            ny = image.shape[1]

            bbox = get_static_map_bounds(lat, lon, 20, nx, ny)
            geotiff_path = save_geotiff_directory + "\\" + image_filename + ".tif"
            save_as_geotif(bbox, image, geotiff_path)

        else:
            print('image with id ' + image_filename + ' does not exist')


def open_geotiff(file_path):
    """
        Function to open a GEOTIFF

        Inputs
        ----------
        file_path : string
            string to GEOTIFF file path

        Outputs
        ----------
        image : numpy array
            RGB image as a numpy array with shape [x_pixels, y_pixels, 3]

        image_bbox : list
            bounding box coordinates of image: [x_min, y_min, x_max, y_max]

        coordinate_system : string
            string with name of coordinate system of coordinates in image_bbox

        """

    # load  image
    raster_src = gdal.Open(file_path, gdal.GA_ReadOnly)

    # get image data and rearrange to get a numpy array with RGB image shape
    data = raster_src.ReadAsArray()
    image = np.dstack((data[0, :, :], data[1, :, :], data[2, :, :]))

    # get image bounding box from geotiff
    ulx, xres, xskew, uly, yskew, yres = raster_src.GetGeoTransform()  # coordinates of upper left corner and resolution
    lrx = ulx + (raster_src.RasterXSize * xres)  # coordinates of lower right corner
    lry = uly + (raster_src.RasterYSize * yres)  # coordinates of lower right corner
    image_bbox = [ulx, lry, lrx, uly]

    # get string of spatial reference system
    src = osr.SpatialReference()
    src.ImportFromWkt(raster_src.GetProjection())
    coordinate_system = (src.GetAttrValue('geogcs'))

    return image, image_bbox, coordinate_system