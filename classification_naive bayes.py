import numpy as np
from osgeo import gdal
import pandas as pd
import config
import os
import matplotlib.pyplot as plt
import cv2
from collections import Counter
# before classifcation, read tif data
raster = gdal.Open('slab1_202001_smallPart.tif')
raster_geotrans = raster.GetGeoTransform() 
raster_proj = raster.GetProjection()
x_size = raster.RasterXSize
y_size = raster.RasterYSize
bandr = raster.GetRasterBand(3)
r = bandr.ReadAsArray(0, 0, 22002, 11749)
bandg = raster.GetRasterBand(2)
g = bandr.ReadAsArray(0, 0, 22002, 11749)
bandb = raster.GetRasterBand(1)
b = bandr.ReadAsArray(0, 0, 22002, 11749)
r = r.reshape((1, 258501498))
g = g.reshape((1, 258501498))
b = b.reshape((1, 258501498))
ori1 = np.row_stack((r, g))
ori2 = np.row_stack((ori1, b))
ori3 = np.transpose(ori2)
print(ori3)
# img2 = cv2.merge([r, g, b])
# plt.imshow(img2)
# plt.xticks([]), plt.yticks([])
# plt.show()
# raster = raster.ReadAsArray(0, 0, x_size, y_size)[0:4]  # you will get a result of array type
# print(raster.RasterCount)
# print(x_size)
# print(y_size)
# print(r.max())
classified = gdal.Open('slab1_smallPart.tif')
x_size2 = classified.RasterXSize
y_size2 = classified.RasterYSize
label = classified.ReadAsArray(0, 0, x_size2, y_size2)
ori_label = label.reshape((1, 258501498))
ori_label = np.transpose(ori_label)
ori4 = np.column_stack((ori3, ori_label))
num1 = np.sum(label == 1)
num2 = np.sum(label == 2)
num3 = np.sum(label == 3)
num4 = np.sum(label == 4)
num5 = np.sum(label == 5)
num15 = np.sum(label == 15)
sum1to15 = 258501498

# label2 = label.astype(np.float)
# result = pd.value_counts(label2)
# print("result:", result)
# your classification
test = gdal.Open('slab4_202001_smallPart.tif')
testr = raster.GetRasterBand(3)
tr = bandr.ReadAsArray(0, 0, 2000, 2000)
testg = raster.GetRasterBand(2)
tg = bandr.ReadAsArray(0, 0, 2000, 2000)
testb = raster.GetRasterBand(1)
tb = bandr.ReadAsArray(0, 0, 2000, 2000)
tr = np.array(tr)
tr2 = tr.reshape((1, 4000000))
tg = np.array(tg)
tg2 = tr.reshape((1, 4000000))
tb = np.array(tb)
tb2 = tr.reshape((1, 4000000))
step1 = np.row_stack((tr2, tg2))
step2 = np.row_stack((step1, tb2))
step3 = np.transpose(step2)
new = np.zeros(4000000)
new = np.transpose(new)
step4 = np.column_stack((step3, new))


def asvoid(arr):
    arr = np.ascontiguousarray(arr)
    return arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[-1])))


def in1d_index(a, b):
    voida, voidb = map(asvoid, (a, b))
    return np.where(np.in1d(voidb, voida))[0]


for i in range(4000000):
    a = step3[i]
    e = [step3[i]]
    z1 = [1]
    z2 = [2]
    z3 = [3]
    z4 = [4]
    z5 = [5]
    z15 = [15]
    a1 = np.column_stack((e,z1))
    a2 = np.column_stack((e,z2))
    a3 = np.column_stack((e,z3))
    a4 = np.column_stack((e,z4))
    a5 = np.column_stack((e,z5))
    a15 = np.column_stack((e,z15))
    b = len(in1d_index(e, ori3))
    b1 = len(in1d_index(a1, ori4))
    b2 = len(in1d_index(a2, ori4))
    b3 = len(in1d_index(a3, ori4))
    b4 = len(in1d_index(a4, ori4))
    b5 = len(in1d_index(a5, ori4))
    b15 = len(in1d_index(a15, ori4))
    pro1 = (b1*num1)/(b*sum1to15)
    pro2 = (b2*num2)/(b*sum1to15)
    pro3 = (b3*num3)/(b*sum1to15)
    pro4 = (b4*num4)/(b*sum1to15)
    pro5 = (b5*num5)/(b*sum1to15)
    pro15 = (b15*num15)/(b*sum1to15)
    maxpro = max(pro1, pro2, pro3, pro4, pro5, pro15)
    if maxpro == pro1:
        step4[i][3] = 1
    if maxpro == pro2:
        step4[i][3] = 2
    if maxpro == pro3:
        step4[i][3] = 3
    if maxpro == pro4:
        step4[i][3] = 4
    if maxpro == pro5:
        step4[i][3] = 5
    if maxpro == pro15:
        step4[i][3] = 15

classified_result = step4[:, 3]
result_matrix = classified_result.reshape((2000, 2000))
np.save("result_matrix.npy", result_matrix)
