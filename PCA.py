#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:15:19 2020

@author: nickwang
"""

import numpy as np
import MNIST
import cv2

def imshow(img):
    cv2.imshow("img", img)
    cv2.waitKey()
cv2.destroyAllWindows()

train_data, train_labels, test_data, test_labels = MNIST.load()

img = train_data[0]
img_avg = np.average(img, axis=(0, 1))
img_std = np.std(img, axis=(0, 1))
img_norm = (img - img_avg) / img_std
img_cov = np.zeros((3, 3))
for data in img_norm.reshape(-1, 3):
    img_cov += data.reshape(3, 1) * data.reshape(1, 3)
img_cov /= len(img_norm.reshape(-1, 3))

eig_values, eig_vectors = np.linalg.eig(img_cov)
for _ in range(100):
    alphas = np.random.normal(0, 0.1, 3)
    img_reconstruct_norm = img_norm + np.sum((eig_values + alphas) * eig_vectors, axis=1)
    img_reconstruct = img_reconstruct_norm * img_std + img_avg
    img_reboundary = np.maximum(np.minimum(img_reconstruct , 255), 0).astype(np.uint8)
    imshow(np.concatenate([img_reboundary, img],axis=0))
imshow(img)

train_data_cov = np.zeros((28*28, 28*28))
for data in train_data:
    train_data_cov += data.reshape(28*28, 1) * data.reshape(1, 28*28)
train_data_cov /= len(train_data)

sub_eig_vectors = eig_vectors[:, :700]
weights = np.dot(train_data[0].reshape(1, 28*28), sub_eig_vectors)
img = np.dot(weights, sub_eig_vectors.T).reshape(28, 28)
img_resize = cv2.resize(img, (img.shape[1] * 30, img.shape[0] * 30))
imshow(img_resize)

def PCA(img):
    img_avg = np.average(img, axis=(0,1))
    img -= img_avg
    train_data_cov = np.zeros((28*28, 28*28))
    for data in train_data:
        train_data_cov += data.reshape(28*28, 1) * data.reshape(1, 28*28)
    train_data_cov /= len(train_data)
