# Import scikit-learn metrics module for accuracy calculation
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
xtrain = xtrain[:, 2000:4000, 5000:7000]
# print(xxtrain.shape)  # (3, 2000, 2000)


ytrain = gdal.Open(r'ytrain_slab1_smallPart.tif')
x_size_ytrain = ytrain.RasterXSize
y_size_ytrain = ytrain.RasterYSize
print(x_size_ytrain)  # 22002
print(y_size_ytrain)  # 11749
ytrain = ytrain.ReadAsArray(0, 0, x_size_ytrain, y_size_ytrain)  # [0:1]
print(set(ytrain.flatten()))  # {1,2,3,4,5,15}
print(ytrain.shape)  # (11749, 22002)
ytrain = ytrain[2000:4000, 5000:7000]

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

# xtest1 = xtest[:, :5000, :5000]
# xtest2 = xtest[:, 5000:10000, :5000]
# xtest3 = xtest[:, 10000:13201, :5000]

# xtest4 = xtest[:, :5000, 5000:10000]
# xtest5 = xtest[:, 5000:10000, 5000:10000]
# xtest6 = xtest[:, 10000:13201, 5000:10000]

# xtest7 = xtest[:, :5000, 10000:15000]
xtest8 = xtest[:, 5000:7000, 10000:12000]
# xtest9 = xtest[:, 10000:13201, 10000:15000]

# xtest10 = xtest[:, :5000, 15000:20000]
# xtest11 = xtest[:, 5000:10000, 15000:20000]
# xtest12 = xtest[:, 10000:13201, 15000:20000]

# xtest13 = xtest[:, :5000, 20000:22364]
# xtest14 = xtest[:, 5000:10000,  20000:22364]
# xtest15 = xtest[:, 10000:13201,  20000:22364]

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
# ytest1 = ytest[:5000, :5000]
# ytest2 = ytest[5000:10000, :5000]
# ytest3 = ytest[10000:13201, :5000]

# ytest4 = ytest[:5000, 5000:10000]
# ytest5 = ytest[5000:10000, 5000:10000]
# ytest6 = ytest[10000:13201, 5000:10000]

# ytest7 = ytest[:5000, 10000:15000]
ytest8 = ytest[5000:7000, 10000:12000]
# ytest9 = ytest[10000:13201, 10000:15000]

# ytest10 = ytest[:5000, 15000:20000]
# ytest11 = ytest[5000:10000, 15000:20000]
# ytest12 = ytest[10000:13201, 15000:20000]

# ytest13 = ytest[:5000, 20000:22364]
# ytest14 = ytest[5000:10000,  20000:22364]
# ytest15 = ytest[10000:13201,  20000:22364]


# convert xtrain size to be (11749*22002,3)
# convert ytrain size to be (11749*22002,1)
# change xtrain to be 11749*22002
xtrainf = np.hsplit(np.dstack(xtrain).flatten(), len(ytrain.flatten()))
ytrainf = ytrain.flatten()

# xtestf1 = np.hsplit(np.dstack(xtest1).flatten(), len(ytest1.flatten()))
# ytestf1 = ytest1.flatten()

# xtestf2 = np.hsplit(np.dstack(xtest2).flatten(), len(ytest2.flatten()))
# ytestf2 = ytest2.flatten()

# xtestf3 = np.hsplit(np.dstack(xtest3).flatten(), len(ytest3.flatten()))
# ytestf3 = ytest3.flatten()

# xtestf4 = np.hsplit(np.dstack(xtest4).flatten(), len(ytest4.flatten()))
# ytestf4 = ytest4.flatten()

# xtestf5 = np.hsplit(np.dstack(xtest5).flatten(), len(ytest5.flatten()))
# ytestf5 = ytest5.flatten()

# xtestf6 = np.hsplit(np.dstack(xtest6).flatten(), len(ytest6.flatten()))
# ytestf6 = ytest6.flatten()

# xtestf7 = np.hsplit(np.dstack(xtest7).flatten(), len(ytest7.flatten()))
# ytestf7 = ytest7.flatten()

xtestf8 = np.hsplit(np.dstack(xtest8).flatten(), len(ytest8.flatten()))
ytestf8 = ytest8.flatten()

# xtestf9 = np.hsplit(np.dstack(xtest9).flatten(), len(ytest9.flatten()))
# ytestf9 = ytest9.flatten()

# xtestf10 = np.hsplit(np.dstack(xtest10).flatten(), len(ytest10.flatten()))
# ytestf10 = ytest10.flatten()

# xtestf11 = np.hsplit(np.dstack(xtest11).flatten(), len(ytest11.flatten()))
# ytestf11 = ytest11.flatten()

# xtestf12 = np.hsplit(np.dstack(xtest12).flatten(), len(ytest12.flatten()))
# ytestf12 = ytest12.flatten()

# xtestf13 = np.hsplit(np.dstack(xtest13).flatten(), len(ytest13.flatten()))
# ytestf13 = ytest13.flatten()

# xtestf14 = np.hsplit(np.dstack(xtest14).flatten(), len(ytest14.flatten()))
# ytestf14 = ytest14.flatten()

# xtestf15 = np.hsplit(np.dstack(xtest15).flatten(), len(ytest15.flatten()))
# ytestf15 = ytest15.flatten()

#####
# Decision tree
print("Starting random forest classification fitting...")
# classifier = DecisionTreeClassifier(criterion='entropy')
# classifier.fit(xtrainf, ytrainf)
# random forest
classifier = RandomForestClassifier()
classifier.fit(xtrainf, ytrainf)

# print decision tree
# fig = plt.figure(figsize=(50, 50))
# _ = tree.plot_tree()

# # predicting the result on the training set using decision tree
print("Starting random forest prediction...")
y_train_pred = classifier.predict(xtrainf)


# y_test_pred1 = classifier.predict(xtestf1)
# y_test_pred2 = classifier.predict(xtestf2)
# y_test_pred3 = classifier.predict(xtestf3)
# y_test_pred4 = classifier.predict(xtestf4)
# y_test_pred5 = classifier.predict(xtestf5)
# y_test_pred6 = classifier.predict(xtestf6)
# y_test_pred7 = classifier.predict(xtestf7)
y_test_pred8 = classifier.predict(xtestf8)
# y_test_pred9 = classifier.predict(xtestf9)
# y_test_pred10 = classifier.predict(xtestf10)
# y_test_pred11 = classifier.predict(xtestf11)
# y_test_pred12 = classifier.predict(xtestf12)
# y_test_pred13 = classifier.predict(xtestf13)
# y_test_pred14 = classifier.predict(xtestf14)
# y_test_pred15 = classifier.predict(xtestf15)

# combine all test result together

# print("The shape of y test1 is : ")
# print(y_test_pred1.shape)
# print(y_test_pred1.reshape((5000, 5000)))
# print(y_test_pred1)

# print("The shape of y test15 is : ")
# print(y_test_pred15.shape)
# print(y_test_pred15.reshape((3201, 2364)))
# print(y_test_pred15)
# y_train_pred_2 = classifier_2.predict(xtrain)
# y_test_pred_2 = classifier_2.predict(xtest)

print("Starting evaluate decision tree model performance...")


def success_ratio(cm):
    total = cm[0][0] + cm[1][0] + cm[0][1] + cm[1][1]
    return 100*(cm[0][0] + cm[1][1]) / total


# evaluate performance
cm_train = confusion_matrix(ytrainf, y_train_pred)
cm_test = confusion_matrix(ytestf8, y_test_pred8)

# cm_test = []
# cm_test.append(confusion_matrix(ytestf1, y_test_pred1))
# cm_test.append(confusion_matrix(ytestf2, y_test_pred2))
# cm_test.append(confusion_matrix(ytestf3, y_test_pred3))
# cm_test.append(confusion_matrix(ytestf4, y_test_pred4))
# cm_test.append(confusion_matrix(ytestf5, y_test_pred5))
# cm_test.append(confusion_matrix(ytestf6, y_test_pred6))
# cm_test.append(confusion_matrix(ytestf7, y_test_pred7))
# cm_test.append(confusion_matrix(ytestf8, y_test_pred8))
# cm_test.append(confusion_matrix(ytestf9, y_test_pred9))
# cm_test.append(confusion_matrix(ytestf10, y_test_pred10))
# cm_test.append(confusion_matrix(ytestf11, y_test_pred11))
# cm_test.append(confusion_matrix(ytestf12, y_test_pred12))
# cm_test.append(confusion_matrix(ytestf13, y_test_pred13))
# cm_test.append(confusion_matrix(ytestf14, y_test_pred14))
# cm_test.append(confusion_matrix(ytestf15, y_test_pred15))


print("Training set confusion matrix : \n"+str(cm_train))
print("Test set confusion matrix : \n"+str(cm_test))
# print("Success ratio on training set : "+str(success_ratio(cm=cm_train))+"%")
# for i in range(15):
#     print("Test set confusion matrix " + str(i) + " is : \n" + str(cm_test[i]))


# after classification, save y_test_pred to tif
# after classification, you need to save your classification result to tif
# print("Starting combining subarray to the whole array...")
# y_test_pred1 = y_test_pred1.reshape((5000, 5000))
# y_test_pred2 = y_test_pred2.reshape((5000, 5000))
# y_test_pred3 = y_test_pred3.reshape((3201, 5000))

# y_test_pred4 = y_test_pred4.reshape((5000, 5000))
# y_test_pred5 = y_test_pred5.reshape((5000, 5000))
# y_test_pred6 = y_test_pred6.reshape((3201, 5000))

# y_test_pred7 = y_test_pred7.reshape((5000, 5000))
y_test_pred8 = y_test_pred8.reshape((2000, 2000))
# y_test_pred9 = y_test_pred9.reshape((3201, 5000))

# y_test_pred10 = y_test_pred10.reshape((5000, 5000))
# y_test_pred11 = y_test_pred11.reshape((5000, 5000))
# y_test_pred12 = y_test_pred12.reshape((3201, 5000))

# y_test_pred13 = y_test_pred13.reshape((5000, 2364))
# y_test_pred14 = y_test_pred14.reshape((5000, 2364))
# y_test_pred15 = y_test_pred15.reshape((3201, 2364))

# # stacking along cols
# y_test_pred_col1 = np.vstack((y_test_pred1, y_test_pred2, y_test_pred3))
# y_test_pred_col2 = np.vstack((y_test_pred4, y_test_pred5, y_test_pred6))
# y_test_pred_col3 = np.vstack((y_test_pred7, y_test_pred8, y_test_pred9))
# y_test_pred_col4 = np.vstack((y_test_pred10, y_test_pred11, y_test_pred12))
# y_test_pred_col5 = np.vstack((y_test_pred13, y_test_pred14, y_test_pred15))
# print("y_test_pred_col1: " + str(y_test_pred_col1.shape))  # (13201, 5000)
# print("y_test_pred_col2: " + str(y_test_pred_col2.shape))  # (13201, 5000)
# print("y_test_pred_col3: " + str(y_test_pred_col3.shape))  # (13201, 5000)
# print("y_test_pred_col4: " + str(y_test_pred_col4.shape))  # (13201, 5000)
# print("y_test_pred_col5: " + str(y_test_pred_col5.shape))  # (13201, 2364)

# # stacking along rows
# y_test_pred = np.hstack((y_test_pred_col1, y_test_pred_col2,
#                         y_test_pred_col3, y_test_pred_col4, y_test_pred_col5))

# print("y_test_pred shape is : " + str(y_test_pred.shape))

print("Starting saving final result into tif...")
driver = gdal.GetDriverByName("GTiff")
path = "randomForestPartial.tif"
cols = 2000  # 22364  # x_size_xtest
rows = 2000  # 13201  # y_size_xtest
dataset = driver.Create(path, cols, rows, 1, gdal.GDT_Float32)
if dataset is not None:
    dataset.SetGeoTransform(raster_geotrans)
    dataset.SetProjection(raster_proj)
    dataset.GetRasterBand(1).WriteArray(y_test_pred8)

# cm_train = confusion_matrix(ytrain, y_train_pred_2)
# cm_test = confusion_matrix(ytest, y_test_pred_2)

# print("Training set confusion matrix : \n"+str(cm_train))
# print("Success ratio on training set : "+str(success_ratio(cm=cm_train))+"%")
# print("Test set confusion matrix : \n"+str(cm_test))
# print("Success ratio on test set : "+str(success_ratio(cm=cm_test))+"%")
