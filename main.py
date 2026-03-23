import geopandas as gpd
import numpy as np
import pandas as pd
import os
import rasterio
import json
import cv2
from skimage.filters import threshold_multiotsu

from skimage.measure import label, regionprops
import json


def convert_to_rgb(src):
    # src = rasterio.open(src)
    img = src.read()
    img = img.transpose([1,2,0])
    img = img.astype('uint8')
    return img

def rgb_to_binarize(img):
    img = cv2.GaussianBlur(img,(3,3),0)
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]
    # convert borders into background values
    G = np.where(G == 0,255,G)
    # find the best thresholds
    thresh = threshold_multiotsu(G)
    # binarization
    binary = G < thresh[0]
    binary = binary.astype('uint8')
    return binary

def morphogical_operations(binary):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

    horizontal = cv2.dilate(binary, kernel, iterations = 3)
    horizontal = cv2.erode(horizontal, kernel,  iterations = 6)
    horizontal = cv2.dilate(horizontal, kernel, iterations = 4)
    horizontal = cv2.erode(horizontal, kernel,  iterations = 6)
    
    return horizontal

def objects_to_dataframe(binary, transform, crs):
    label_image = label(binary)
    ls_x = []
    ls_y = []
    for region in regionprops(label_image):
        if region.area >= 1:
            y0, x0 = region.centroid
            xs, ys = rasterio.transform.xy(transform, y0, x0)
            ls_x.append(xs)
            ls_y.append(ys)
    df_xy = pd.DataFrame([])
    df_xy['x'] = ls_x
    df_xy['y'] = ls_y
    gdf = gpd.GeoDataFrame(df_xy, geometry=gpd.points_from_xy(df_xy['x'], df_xy['y']))
    gdf = gdf.set_crs(crs)
    
    return gdf

def get_homogeneity_index(binary):
    num_labels, labels_im = cv2.connectedComponents(binary)
    # ignora fundo (label 0)
    areas = np.bincount(labels_im.ravel())[1:]

    mean_area = np.mean(areas)
    std_area = np.std(areas)

    cv = std_area / mean_area
    # print("Indice de homogeneidade: ", cv)
    uniformity = 1 / (1 + cv)
    # print("Indice de homogeneidade amigavel: ", uniformity) # between 0 and 1
    return uniformity

def gdf_to_json(gdf_plants, area_m2, uniformity, filename):

    total_plants = len(gdf_plants)  # GeoDataFrame com plantas
    area_hectares = area_m2 / 10000  # converter m² → hectares

    result = {
        "total_plants": total_plants,
        "area_hectares": area_hectares,
        "plants_per_hectare_avg": total_plants / area_hectares if area_hectares > 0 else 0,
        "homogeneity_plant_index": uniformity
    }

    with open(f"output/statistic-{filename}.json", "w") as f:
        json.dump(result, f, indent=4)


def main(path_tif):
    with rasterio.open(path_tif) as src:
        pixel_size_x = src.res[0]  # width of pixel (meters)
        pixel_size_y = src.res[1]  # height of pixel (meters)
        
        pixel_area = pixel_size_x * pixel_size_y  # m²
        
        data = src.read(1)
        
        # count valid pixels (exclude nodata)
        if src.nodata is not None:
            valid_pixels = np.sum(data != src.nodata)
        else:
            valid_pixels = data.size

        total_area_m2 = valid_pixels * pixel_area

        img_rgb = convert_to_rgb(src)
        img_binary = rgb_to_binarize(img_rgb)
        img_binary = morphogical_operations(img_binary)
        gdf_objects = objects_to_dataframe(img_binary, src.transform, src.crs.to_dict())
        uniformity_plant_index = get_homogeneity_index(img_binary)
        
        output_filename = os.path.basename(path_tif.split(".")[0])
        gdf_objects.to_file(f"output/objects_{output_filename}.geojson")

        gdf_to_json(gdf_objects, total_area_m2, uniformity_plant_index, output_filename)
        

path_tif = "data/sample1.tif"
main(path_tif)