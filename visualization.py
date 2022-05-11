__author__ = "Sebastian Krapf"
__copyright__ = "Copyright 2021, "
__credits__ = []
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Sebastian Krapf"
__email__ = "sebastian.krapf@tum.de"
__status__ = "alpha"

from shapely.geometry import Point, LineString, box, LinearRing, Polygon, MultiPoint, MultiLineString, MultiPolygon, GeometryCollection
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sn
import cv2
import geopandas as gpd
import numpy as np

from model_training import denormalize
from utils import convert_between_latlon_and_pixel

class TUM_CI_colors:
    cmap_name = 'TUM'
    # custom TUM-CI colormap
    colors = [(1, 1, 1), (0, 82 / 255, 147 / 255)]  # R -> G -> B
    TUM_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(cmap_name, colors, N=100)

    # <=========== Main colors ===========>
    TUM_blue = np.divide([0, 101, 189], 255)
    black = np.divide([0, 0, 0], 255)
    white = np.divide([255, 255, 255], 255)

    # <=========== Secondaries ===========>
    dark_blue = np.divide([0, 82, 147], 255)
    light_blue = np.divide([100, 160, 200], 255)
    lighter_blue = np.divide([152, 198, 234], 255)
    gray = np.divide([153, 53, 153], 255)
    # <===========   Emphasis  ===========>
    orange = np.divide([227, 114, 34], 255)
    green = np.divide([162, 173, 0], 255)
    light_gray = np.divide([218, 215, 203], 255)

    # sem_seg_map
    colors = list([TUM_blue, lighter_blue, orange, black, light_blue, light_gray, dark_blue, green, white])
    sem_seg_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('sem_seg_map', colors, N=9)



def cm_to_inch(length_cm):
    length_inch = length_cm*0.3937
    return length_inch

# Define image size
def get_image_size(version):
    if version == 'small':
        img_size_x = cm_to_inch(9.23)
        img_size_y = img_size_x
    elif version == 'wide':
        img_size_x = cm_to_inch(18.46)
        img_size_y = img_size_x/6
    elif version == 'large':
        img_size_x = cm_to_inch(18.46)
        img_size_y = img_size_x / 4
    elif version == 'one_third_large':
        img_size_x = cm_to_inch(18.46)
        img_size_y = img_size_x / 3

    return img_size_x, img_size_y

# Plot a Point
def plot_pt(ax, pt, *args, **kwargs):
    ax.scatter(*pt.xy, *args, **kwargs)

# Plot a LineString
def plot_ls(ax, ls, *args, **kwargs):
    ax.plot(*ls.xy, *args, **kwargs)

# Plot a LinearRing
def plot_lr(ax, lr, *args, **kwargs):
    plot_ls(ax, lr, *args, **kwargs)
    
# Plot a Polygon
def plot_poly(ax, poly, *args, **kwargs):
    ax.plot(*poly.exterior.xy, *args, **kwargs)
    for pol_in in poly.interiors:
        ax.plot(*pol_in.xy, *args, **kwargs)
        
# Plot a MultiPoint
def plot_mpt(ax, mpt, *args, **kwargs):
    for pt in mpt:
        plot_pt(ax, pt, *args, **kwargs)
        
# Plot a MultiLineString
def plot_mls(ax, mls, *args, **kwargs):
    for ls in mls:
        plot_ls(ax, ls, *args, **kwargs)
        
# Plot a MultiPolygon
def plot_mpoly(ax, mpoly, *args, **kwargs):
    for poly in mpoly:
        plot_poly(ax, poly, *args, **kwargs)
    
# Plot a GeometryCollection
def plot_gc(ax, gc, *args, **kwargs):
    for geom in gc:
        if type(geom) == Point:
            plot_pt(ax, geom, *args, **kwargs)
        elif type(geom) == LineString or type(geom) == LinearRing:
            plot_ls(geom, *args, **kwargs)
        elif type(geom) == Polygon:
            plot_poly(ax, geom, *args, **kwargs)
        elif type(geom) == MultiPolygon:
            plot_mpoly(ax, geom, *args, **kwargs)
        elif type(geom) == MultiPoint:
            plot_mpt(ax, geom, *args, **kwargs)
        elif type(geom) == MultiLineString:
            plot_mls(ax, geom, *args, **kwargs)
        elif type(geom) == GeometryCollection:
            plot_gc(ax, geom, *args, **kwargs)
        else:    
            raise Exception("Unknown geometry type in GeometryCollection! Can't plot")


def plot_shp(ax, geom, *args, **kwargs):
    print(geom)
    if type(geom) == Point:
        plot_pt(ax, geom, *args, **kwargs)
    elif type(geom) in [LineString, LinearRing]:
        plot_ls(ax, geom, *args, **kwargs)
    elif type(geom) in [Polygon, box]:
        plot_poly(ax, geom, *args, **kwargs)
    elif type(geom) == MultiLineString:
        plot_mls(ax, geom, *args, **kwargs)
    elif type(geom) == MultiPoint:
        plot_mpt(ax, geom, *args, **kwargs)
    elif type(geom) == MultiPolygon:
        plot_mpoly(ax, geom, *args, **kwargs)
    elif type(geom) == GeometryCollection:
        plot_gc(ax, geom, *args, **kwargs)
    else:
        raise Exception("Unknown geometry type! Can't plot")


def plot_train_val_test_data(gdf_train_image_list, gdf_val_image_list, gdf_test_images, gdf_test_images_all):
    # variate center coordinate to create different train-val data set splits
    fig, ax = plt.subplots(figsize=(get_image_size('small')))
    gdf_train_image_list[0].plot(alpha=0.3, color=TUM_CI_colors.green, ax=ax)
    gdf_val_image_list[0].plot(alpha=0.3, color=TUM_CI_colors.green, ax=ax)

    for i, gdf_val_image in enumerate(gdf_val_image_list):
        col = TUM_CI_colors.TUM_cmap(255 / len(gdf_val_image_list))
        gdf_val_image.plot(alpha=0.1, color=col, ax=ax)
        plot_shp(ax, gdf_val_image.geometry.unary_union.convex_hull, color=col, alpha=0.3)

    gdf_test_images.plot(alpha=0.8, color=TUM_CI_colors.orange, ax=ax)
    gdf_test_images_all.plot(alpha=0.8, color=TUM_CI_colors.gray, ax=ax)


def box_plot_IoU_TUM_CI(data, data_label_list, save_path, colors=[], image_size='small', show_means=True):
    fig, ax = plt.subplots(figsize=(get_image_size(image_size)))
    fig.tight_layout()

    bp = ax.boxplot(data,
                    showcaps=False,
                    widths=0.3,
                    patch_artist=True,
                    showmeans=show_means,
                    showfliers=True,
                    meanline=True,
                    meanprops=dict(linestyle='dashed'),
                    flierprops=dict(marker='o', markerfacecolor=TUM_CI_colors.black, markersize=1))

    if len(colors)==0:
        colors = [TUM_CI_colors.white for i in enumerate(data)]

    for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps', 'means']:
        plt.setp(bp[element], color=TUM_CI_colors.black)
    for patch, color in zip(bp['boxes'], colors):
        patch.set(facecolor=color)
    for patch in bp['means']:
        patch.set(color=TUM_CI_colors.orange)
    for patch in bp['medians']:
        patch.set(color=TUM_CI_colors.TUM_blue)

    plt.ylim(0, 1)
    plt.xticks(np.arange(1, len(data_label_list)+1), data_label_list)
    Lines1 = Line2D([0], [0], color=TUM_CI_colors.TUM_blue, linewidth=1, linestyle='-')
    Lines2 = Line2D([0], [0], color=TUM_CI_colors.orange, linewidth=1, linestyle='dashed')
    labels = ['median', 'mean']
    Lines = [Lines1, Lines2]
    L = plt.legend(Lines, labels)
    for tick in ax.get_xticklabels():
        tick.set_fontname("Palatino Linotype")
        tick.set_fontsize(9)
    for tick in ax.get_yticklabels():
        tick.set_fontname("Palatino Linotype")
        tick.set_fontsize(9)
    plt.tick_params(bottom=False)
    plt.ylabel('IoU', fontname="Palatino Linotype", fontsize=9)
    plt.setp(L.texts, family="Palatino Linotype", fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    plt.savefig(save_path, format='svg', bbox_inches='tight')

    return fig, ax


def box_plot_TUM_CI_annotation_and_prediction(roof, annotation, prediction, all_ticks_list,
                                              save_path, image_size='small', show_means=True):
    fig, ax = plt.subplots(figsize=(get_image_size(image_size)))
    fig.tight_layout()

    bp_roof = ax.boxplot(roof,
                         positions=[0],
                         showcaps=False,
                         widths=0.3,
                         patch_artist=True,
                         showmeans=show_means,
                         showfliers=True,
                         meanline=True,
                         meanprops=dict(linestyle='dashed'),
                         boxprops=dict(facecolor=TUM_CI_colors.white),
                         flierprops=dict(marker='o', markerfacecolor=TUM_CI_colors.black, markersize=1))

    bp_annotation = ax.boxplot(annotation,
                               positions=np.arange(1,2*len(annotation),2),
                               showcaps=False,
                               widths=0.3,
                               patch_artist=True,
                               showmeans=show_means,
                               showfliers=True,
                               meanline=True,
                               meanprops=dict(linestyle='dashed'),
                               boxprops=dict(facecolor=TUM_CI_colors.light_blue),
                               flierprops=dict(marker='o', markerfacecolor=TUM_CI_colors.black, markersize=1))

    bp_prediction = ax.boxplot(prediction,
                               positions=np.arange(2, 2*len(prediction)+1, 2),
                               showcaps=False,
                               widths=0.3,
                               patch_artist=True,
                               showmeans=show_means,
                               showfliers=True,
                               meanline=True,
                               meanprops=dict(linestyle='dashed'),
                               boxprops=dict(facecolor=TUM_CI_colors.light_gray),
                               flierprops=dict(marker='o', markerfacecolor=TUM_CI_colors.black, markersize=1))

    # for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps', 'means']:
    #     plt.setp(bp[element], color=TUM_CI_colors.black)
    # for patch in bp['means']:
    #     patch.set(color=TUM_CI_colors.orange)
    # for patch in bp['medians']:
    #     patch.set(color=TUM_CI_colors.TUM_blue)

    plt.ylim(0, 1)
    plt.xticks(np.arange(0, len(all_ticks_list)), all_ticks_list)

    Lines1 = Line2D([0], [0], color=TUM_CI_colors.orange, linewidth=1, linestyle='-')
    Lines2 = Line2D([0], [0], color=TUM_CI_colors.green, linewidth=1, linestyle='dashed')
    Lines = [Lines1, Lines2]

    ax.legend([bp_roof["boxes"][0], bp_annotation["boxes"][0], bp_prediction["boxes"][0], Lines1, Lines2],
              ['roof outline', 'annotations', 'predictions', 'median', 'mean'], loc='upper right', ncol=5, bbox_to_anchor=(0.92, 1.3))

    for tick in ax.get_xticklabels():
        tick.set_fontname("Palatino Linotype")
        tick.set_fontsize(9)
    for tick in ax.get_yticklabels():
        tick.set_fontname("Palatino Linotype")
        tick.set_fontsize(9)
    plt.tick_params(bottom=False)
    plt.ylabel('IoU', fontname="Palatino Linotype", fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    plt.savefig(save_path, format='svg', bbox_inches='tight')

    return fig, ax



def confusion_matrix_heatmap_TUM_CI(data, data_label_list, save_path, image_size='small'):
    plt.rcParams["font.family"] = "Palatino Linotype"
    plt.rcParams["font.size"] = "9"
    tick_labels = data_label_list
    fig, ax = plt.subplots(figsize=(get_image_size(image_size)))
    fig.tight_layout()
    ax = sn.heatmap(data,
                    vmin=0,
                    vmax=1,
                    cbar=False,
                    xticklabels=tick_labels,
                    yticklabels=tick_labels,
                    cmap=TUM_CI_colors.TUM_cmap,
                    annot=np.round(data, 2)
                    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_xlabel("prediction")
    ax.set_ylabel("ground truth")
    plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')

    return fig, ax


def visualize_image_gt_pr(img_list, gt_list, pr_list, label_classes, save_path):
# Define grid layout
    ncols = 6
    nrows = 4
    height_ratio_list = [n for n in np.repeat([1/nrows], nrows-1)]
    height_ratio_list.append(0.13)
    # height_ratio_list.append(0.1)
    width_ratio_list = [n for n in np.repeat([1/ncols], ncols)]
    # width_ratio_list.append(0.1)

    fig = plt.figure(constrained_layout=True, figsize=(6.858, 3.429*1.13))
    fig.tight_layout()

    spec = gridspec.GridSpec(ncols=ncols,
                             nrows=nrows,
                             figure=fig,
                             height_ratios=height_ratio_list,
                             width_ratios=width_ratio_list)

    axes = []
    for i in np.arange(0, ncols):
        for j in np.arange(0, nrows-1):
            ax = fig.add_subplot(spec[j, i])
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            axes.append(ax)
    lax = fig.add_subplot(spec[3, :])
    lax.xaxis.set_ticks([])
    lax.yaxis.set_ticks([])
    lax.axis("off")

    # plot image, ground truth and prediction
    for i in np.arange(0, ncols):
        axes[3*i].imshow(denormalize(img_list[i].squeeze()))
        axes[3*i+1].imshow(gt_list[i], cmap=TUM_CI_colors.sem_seg_cmap, clim=[0, len(label_classes)])
        axes[3*i+2].imshow(pr_list[i], cmap=TUM_CI_colors.sem_seg_cmap, clim=[0, len(label_classes)])

    class_list = list(label_classes) + ['background']
    values = np.arange(0, len(class_list))
    colors = [ TUM_CI_colors.sem_seg_cmap(value) for value in values]
    # create a patch (proxy artist) for every color
    patches = [mpatches.Patch(color=colors[i], label=class_list[i] ) for i in range(len(values))]
    fig.tight_layout(pad=0)
    lax.legend(handles=patches, loc='center', bbox_to_anchor=(0.5, 0.65),  ncol=5)


    plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')

    return


def visualize_labels():
    dir_images = 'data\\images_roof_centered_png\\'
    dir_superstructure_masks = 'data\\masks_superstructures_initial\\'
    dir_segments_masks = 'data\\masks_segments\\'
    img_files = ['244.png', '373.png']

    fig = plt.figure(constrained_layout=True, figsize=(4.6, 3))

    ncols = 3
    nrows = 2
    height_ratio_list = [n for n in np.repeat([1 / nrows], nrows)]
    width_ratio_list = [n for n in np.repeat([1 / ncols], ncols)]

    spec = gridspec.GridSpec(ncols=ncols,
                             nrows=nrows,
                             figure=fig,
                             height_ratios=height_ratio_list,
                             width_ratios=width_ratio_list)

    axes = []
    for i in np.arange(0, ncols):
        for j in np.arange(0, nrows):
            ax = fig.add_subplot(spec[j, i])
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            axes.append(ax)

    image = cv2.imread(dir_images + img_files[0])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    axes[0].imshow(image)
    image = cv2.imread(dir_images + img_files[1])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    axes[1].imshow(image)

    axes[2].imshow(cv2.imread(dir_segments_masks + img_files[0], 0), cmap=TUM_CI_colors.TUM_cmap.reversed())
    axes[3].imshow(cv2.imread(dir_segments_masks + img_files[1], 0), cmap=TUM_CI_colors.TUM_cmap.reversed())
    axes[4].imshow(cv2.imread(dir_superstructure_masks + img_files[0], 0), cmap=TUM_CI_colors.TUM_cmap.reversed())
    axes[5].imshow(cv2.imread(dir_superstructure_masks + img_files[1], 0), cmap=TUM_CI_colors.TUM_cmap.reversed())

    fig.tight_layout(pad=0.02)

    plt.savefig('plots\\eye_catcher.svg', format='svg', dpi=300, bbox_inches='tight')

    return


def box_plot_E_gen_TUM_CI(data, data_label_list, E_sums, y_label, save_path, colors=TUM_CI_colors.light_blue,
                          image_size='wide', show_means=True):
    fig, ax = plt.subplots(figsize=(get_image_size(image_size)))
    fig.tight_layout()

    bp = ax.boxplot(data,
                    showcaps=False,
                    widths=0.3,
                    patch_artist=True,
                    showmeans=show_means,
                    showfliers=True,
                    meanline=True,
                    meanprops=dict(linestyle='dashed'),
                    flierprops=dict(marker='o', markerfacecolor=TUM_CI_colors.black, markersize=1))

    if len(colors)==0:
        colors = [TUM_CI_colors.white for i in enumerate(data)]

    for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps', 'means']:
        plt.setp(bp[element], color=TUM_CI_colors.black)
    for patch, color in zip(bp['boxes'], colors):
        patch.set(facecolor=color)
    for patch in bp['means']:
        patch.set(color=TUM_CI_colors.green)
    for patch in bp['medians']:
        patch.set(color=TUM_CI_colors.orange)

    # plt.ylim(0, 1)
    plt.xticks(np.arange(1, len(data_label_list)+1), data_label_list)
    Lines1 = Line2D([0], [0], color=TUM_CI_colors.orange, linewidth=1, linestyle='-')
    Lines2 = Line2D([0], [0], color=TUM_CI_colors.green, linewidth=1, linestyle='dashed')
    labels = ['median', 'mean']
    Lines = [Lines1, Lines2]
    L = plt.legend(Lines, labels)
    for tick in ax.get_xticklabels():
        tick.set_fontname("Palatino Linotype")
        tick.set_fontsize(9)
    for tick in ax.get_yticklabels():
        tick.set_fontname("Palatino Linotype")
        tick.set_fontsize(9)
    plt.tick_params(bottom=False)
    plt.ylabel(y_label, fontname="Palatino Linotype", fontsize=9)
    plt.setp(L.texts, family="Palatino Linotype", fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    # add energy generation sum
    for i, E in enumerate(E_sums):
        ax.text(i + 1, np.max(data) * 1.08, 'sum: ' + str(E) + '\n GWh / a', ha='center',
                fontname="Palatino Linotype",
                fontsize=9)
    fig.tight_layout()

    plt.savefig(save_path, format='svg', bbox_inches='tight')

    return fig, ax



def visualize_module_placement(image_ids, gdf_images, gdf_segments, gdf_superstructures, gdf_modules, DIR_IMAGES):
    fig, axs = plt.subplots(1, len(image_ids), figsize=(get_image_size('one_third_large')))
    fig.tight_layout()

    # transform to crs 4326
    gdf_images = gdf_images.to_crs(4326)
    gdf_segments = gdf_segments.to_crs(4326)
    gdf_superstructures = gdf_superstructures.to_crs(4326)
    gdf_modules = gdf_modules.to_crs(4326)

    # chosen images to plot
    gdf_images_red = gdf_images[gdf_images.id.isin([str(id) for id in image_ids])]

    # image_bbox_px = box(0, 0, 512, 512)
    for i, image_id in enumerate(image_ids):
        gdf_image = gdf_images_red[gdf_images_red.id == image_id]

        ax = axs[i]
        # add image
        image = plt.imread(DIR_IMAGES + "\\" + str(image_id) + '.png')
        # get image bounds
        image_bbox = gdf_image.iloc[0].geometry
        x_min = image_bbox.bounds[0]
        x_max = image_bbox.bounds[2]
        y_min = image_bbox.bounds[1]
        y_max = image_bbox.bounds[3]
        # plot image and set bounds
        ax.imshow(image, extent=[x_min, x_max, y_min, y_max])
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(x_min, x_max)

        # get segments in image
        intersections = gpd.tools.sjoin(gdf_image, gdf_segments, how="inner", op='intersects')
        gdf_segments_ids = intersections.id_right.unique()
        gdf_segments_red = gdf_segments[gdf_segments.id.isin(gdf_segments_ids)]

        # get superstructures in image
        intersections = gpd.tools.sjoin(gdf_image, gdf_superstructures, how="inner", op='intersects')
        gdf_superstructures_ids = intersections.id_right.unique()
        gdf_superstructures_red = gdf_superstructures[gdf_superstructures.id.isin(gdf_superstructures_ids)]
        # get unary union and make new dataframe
        gdf_superstructures_uu = gpd.GeoDataFrame({'geometry': gdf_superstructures_red.unary_union})

        # get modules in image
        intersections = gpd.tools.sjoin(gdf_image, gdf_modules, how="inner", op='intersects')
        gdf_modules_ids = intersections.index_right.unique()
        gdf_modules_red = gdf_modules.iloc[gdf_modules_ids]

        # plot segments, superstructures and modules and show
        gdf_segments_red.plot(ax=ax, alpha=0.2, color=TUM_CI_colors.lighter_blue, edgecolor=TUM_CI_colors.black)
        gdf_superstructures_uu.plot(ax=ax, alpha=0.3, color=TUM_CI_colors.green, edgecolor=TUM_CI_colors.green)
        gdf_modules_red.plot(ax=ax, alpha=0.4, color=TUM_CI_colors.light_blue, edgecolor=TUM_CI_colors.dark_blue)

        ax.axis("off")

    fig.tight_layout()
    plt.savefig('plots\\module_placement_examples_2d_3d.svg', format='svg', dpi=300, bbox_inches='tight')

    return


def segment_polar_plot(label_classes, label_area_count_4, label_area_count_8, label_area_count_16):

    # set figure size
    fig, ax = plt.subplots(figsize=(get_image_size('one_third_large')))
    plt.axis('off')

    # add grid
    ncols = 3
    nrows = 1
    height_ratio_list = [n for n in np.repeat([1 / nrows], nrows)]
    width_ratio_list = [n for n in np.repeat([1 / ncols], ncols)]

    spec = gridspec.GridSpec(ncols=ncols,
                             nrows=nrows,
                             figure=fig,
                             height_ratios=height_ratio_list,
                             width_ratios=width_ratio_list)

    label_area_count_list = list([label_area_count_4, label_area_count_8, label_area_count_16])
    # plot polar axis
    for i in np.arange(0, ncols):
        label_area_count = label_area_count_list[i]
        N = len(label_area_count)-1
        ax = fig.add_subplot(spec[i], projection='polar')

        # remove grid
        ax.axis('on')

        area_without_flat = label_area_count[0:N]

        # Compute the width of each bar. In total we have 2*Pi = 360°
        width = 2 * np.pi / N

        # Compute the angle each bar is centered on:
        indexes = list(range(0, len(area_without_flat)))
        angles = [element * width for element in indexes]
        labels = list(label_classes.values())
        # ax.set_xticks(angles)
        ax.set_xticks(np.pi / 180. * np.linspace(0, 360, 16, endpoint=False))
        ax.set_xticklabels(labels[0:16], fontname="Palatino Linotype", fontsize=9)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.set_yticklabels(["", "", 0.15, "", "", 0.3], fontname="Palatino Linotype", fontsize=9)
        ax.set_ylim([0, 0.3])
        ax.set_rlabel_position(107)
        # Draw bars
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        bars = ax.bar(
            x=angles,
            height=area_without_flat,
            width=width,
            bottom=0,
            linewidth=2,
            edgecolor="white")

    fig.tight_layout(pad=0.5)
    plt.savefig('plots\\segments_polar_plot.svg', format='svg', dpi=300, bbox_inches='tight')

    return


def visualization_annotation_agreement(gdf_annotations, label_classes, annotator_ids, building_id):
    fig, ax = plt.subplots()
    values = np.arange(0, len(label_classes))
    colors = [ TUM_CI_colors.sem_seg_cmap(value) for value in values]

    # select two annotators that should be compared
    if len(annotator_ids) == 0:
        annotator_ids = [1,3]
    gdf_plot = gdf_annotations[gdf_annotations.Labeler.isin(annotator_ids)]

    # select building that should be compared
    if len(building_id) == 0:
        building_id = [5]
    gdf_plot = gdf_plot[gdf_plot.haus.isin(building_id)]

    for annotator in annotator_ids:
        gdf_annotator = gdf_plot[gdf_plot.Labeler.isin([annotator])]
        if annotator == annotator_ids[0]:
            edge_color = TUM_CI_colors.TUM_blue
        elif annotator == annotator_ids[1]:
            edge_color = TUM_CI_colors.orange
        else:
            print("please only select two annotators or change code")

        for label_id, label_class in label_classes.items():
            # For each loop, consider labels of specific class, only
            gdf_plot_class = gdf_annotator[gdf_annotator.class_type == label_class]
            color = colors[label_id]
            if len(gdf_plot_class) != 0:
                gdf_plot_class.plot(ax=ax, color=color, alpha=0.5, edgecolor=edge_color)

    plt.savefig('plots\\annotations.svg', format='svg', dpi=300, bbox_inches='tight')

    return