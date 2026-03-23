import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import math
import os
import rasterio
from rasterio.mask import mask
import json
from rasterio.plot import show
import matplotlib.pyplot as plt
import cv2
from skimage.filters import threshold_multiotsu
import matplotlib.patches as mpatches

from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb


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

def objects_to_dataframe(binary, transform, crs, filename):
    num_labels, labels_im = cv2.connectedComponents(binary)
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
    gdf.to_file(f"output/objects_{filename}.geojson")


def main():
    path_tif = "data/sample2.tif"
    src = rasterio.open(path_tif)
    img_rgb = convert_to_rgb(src)
    img_binary = rgb_to_binarize(img_rgb)
    img_binary = morphogical_operations(img_binary)
    output_filename = os.path.basename(path_tif.split(".")[0])
    objects_to_dataframe(img_binary, src.transform, src.crs.to_dict(), output_filename)

main()
    
