# Import scikit-learn metrics module for accuracy calculation
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn import metrics
# Import train_test_split function
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from numpy import savetxt
from numpy import asarray
import pydotplus  # pip install pydotplus
import seaborn as sns
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import warnings
import numpy as np
import pandas as pd
import config
import os
from osgeo import gdal, ogr
import cv2
import matplotlib.pyplot as plt


xtrain = gdal.Open(r'xtrain_slab1_smallPart.tif')
band1 = xtrain.GetRasterBand(1).ReadAsArray()  # red band
# print(band1)
# plt.imshow(band1, cmap='gray')

gt = xtrain.GetGeoTransform()
# print(gt)
print("\n")
proj = xtrain.GetProjection()
# print(proj)

# check band
# print(xtrain.RasterCount)  # 3

x_size = xtrain.RasterXSize
y_size = xtrain.RasterYSize
print(x_size)  # 22003 width
print(y_size)  # 11750 height
# you will get a result of array type
# label: [0,1]
xtrain = xtrain.ReadAsArray(0, 0, x_size, y_size)[0:3]  # 0, 1, 2 RGB
print(xtrain.shape)  # (3, 11750, 22003)
# change to be (3,11749, 22002)
xtrain = xtrain[:, :-1, :-1]
# xtrain = xtrain[:, 2000:7000, 2000:7000]
# print(xxtrain.shape)  # (3, 2000, 2000)


ytrain = gdal.Open(r'ytrain_slab1_smallPart.tif')
x_size_ytrain = ytrain.RasterXSize
y_size_ytrain = ytrain.RasterYSize
print(x_size_ytrain)  # 22002
print(y_size_ytrain)  # 11749
ytrain = ytrain.ReadAsArray(0, 0, x_size_ytrain, y_size_ytrain)  # [0:1]
print(set(ytrain.flatten()))  # {1,2,3,4,5,15}
print(ytrain.shape)  # (11749, 22002)
# ytrain = ytrain[2000:7000, 2000:7000]

# x test
xtest = gdal.Open(r'xtest_slab4_smallPart.tif')
raster_geotrans = xtest.GetGeoTransform()
raster_proj = xtest.GetProjection()
x_size_xtest = xtest.RasterXSize
y_size_xtest = xtest.RasterYSize
print(x_size_xtest)  # 22364
print(y_size_xtest)  # 13202
xtest = xtest.ReadAsArray(0, 0, x_size_xtest, y_size_xtest)[0:3]  # [0:1]
# print(xtest.shape)  # (3, 13202, 22364)
# change it to be (3,13201,22364)
xtest = xtest[:, :-1, :]

print(xtest.shape)  # (3, 13201,22364)
# print(xtest15.shape)  # (3,3201,2364)


# y test
ytest = gdal.Open(r'ytest_slab4_smallPart.tif')
x_size_ytest = ytest.RasterXSize
y_size_ytest = ytest.RasterYSize
print(x_size_ytest)  # 22364
print(y_size_ytest)  # 13201
ytest = ytest.ReadAsArray(0, 0, x_size_ytest, y_size_ytest)
print(set(ytest.flatten()))  # {1,2,3,4,5,15}
print(ytest.shape)  # (13201, 22364)


# convert xtrain size to be (11749*22002,3)
# convert ytrain size to be (11749*22002,1)
# change xtrain to be 11749*22002
xtrainf = np.hsplit(np.dstack(xtrain).flatten(), len(ytrain.flatten()))
ytrainf = ytrain.flatten()


xtestf = np.hsplit(np.dstack(xtest).flatten(), len(ytest.flatten()))
ytestf = ytest.flatten()


# random forest
classifier = RandomForestClassifier()
classifier.fit(xtrainf, ytrainf)


# # predicting the result on the training set using random forest
print("Starting random forest prediction...")
y_train_pred = classifier.predict(xtrainf)
y_test_pred = classifier.predict(xtestf)

print("Starting evaluate randome forest model performance...")


def success_ratio(cm):
    total = cm[0][0] + cm[1][0] + cm[0][1] + cm[1][1]
    return 100*(cm[0][0] + cm[1][1]) / total


# evaluate performance
cm_train = confusion_matrix(ytrainf, y_train_pred)
cm_test = confusion_matrix(ytestf, y_test_pred)

print("Training set confusion matrix : \n"+str(cm_train))
print("Test set confusion matrix : \n"+str(cm_test))

y_test_pred = y_test_pred.reshape((ytest.shape))  # (13201, 22364)

print("Starting saving final result into tif...")
driver = gdal.GetDriverByName("GTiff")
path = "RandomForestResult.tif"
cols = 22364  # x_size_xtest
rows = 13201  # y_size_xtest
dataset = driver.Create(path, cols, rows, 1, gdal.GDT_Float32)
if dataset is not None:
    dataset.SetGeoTransform(raster_geotrans)
    dataset.SetProjection(raster_proj)
    dataset.GetRasterBand(1).WriteArray(y_test_pred)
