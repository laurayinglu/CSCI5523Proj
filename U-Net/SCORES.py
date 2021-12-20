import sys
sys.path.append("../")
import config
import os
import numpy as np
import torch
from collections import Counter
from osgeo import gdal, gdalconst, osr
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
np.set_printoptions(suppress=True)

print("#######################################################################")
print("MAKE DIRECTORIES")
if not os.path.exists(os.path.join(config.RESULT_DIR, "METRICS", "CONF_MAT")):
    os.makedirs(os.path.join(config.RESULT_DIR, "METRICS", "CONF_MAT"))

# for tile in config.tiles:
tile = config.tiles[0]
for row in range(config.grids):
    for col in range(config.grids):
        grid = tile+"_"+str(row)+"_"+str(col)
        print(grid)
        
        label = np.load(os.path.join(config.NUMPY_DIR, grid+"_label.npy"))
        label_valid = np.load(os.path.join(config.NUMPY_DIR, grid+"_label_valid.npy"))
        pred = np.load(os.path.join(config.RESULT_DIR, "MAPS", "PRED_MAPS", grid+"_pred_map.npy"))
        pred_valid = pred!=config.unknown_class
        
        conf_mat = np.zeros((config.classes, config.classes))
        for label_class_val in range(config.classes):
            for pred_class_val in range(config.classes):
                conf_mat[label_class_val, pred_class_val] += np.sum((label[label_valid & pred_valid]==label_class_val) & (pred[label_valid & pred_valid]==pred_class_val))
                
        np.save(os.path.join(config.RESULT_DIR, "METRICS", "CONF_MAT", grid+"_conf_mat"), conf_mat)

conf_mat = np.zeros((config.classes, config.classes))
# for tile in config.tiles:
for row in range(config.grids):
    for col in range(config.grids):
        grid = tile+"_"+str(row)+"_"+str(col)
        print(grid)
        
        conf_mat += np.load(os.path.join(config.RESULT_DIR, "METRICS", "CONF_MAT", grid+"_conf_mat.npy"))

print(conf_mat)