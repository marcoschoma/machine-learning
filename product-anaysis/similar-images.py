import numpy as np
# import matplotlib.pyplot as plt

from scipy.misc import imread, imresize, imshow
import cv2
import os
import time
from hashlib import md5
import scipy

IMAGE_DIR = '/home/marcos/Downloads/testeImg'

os.chdir(IMAGE_DIR)
os.getcwd()

image_files = os.listdir(IMAGE_DIR)
print(len(image_files))

def filter_images(images):
    image_list = []
    for image in images:
        try:
            assert imread(image).shape[2] == 3
            image_list.append(image)
        except  AssertionError as e:
            print(e)
    return image_list

#First turn the image into a gray scale image
def img_gray(image):
    image = imread(image)
    return np.average(image, weights=[0.299, 0.587, 0.114], axis=2)

#resize image and flatten
def resize(image, height=30, width=30):
    row_res = cv2.resize(image,(height, width), interpolation = cv2.INTER_AREA).flatten()
    col_res = cv2.resize(image,(height, width), interpolation = cv2.INTER_AREA).flatten('F')
    return row_res, col_res

#gradient direction based on intensity 
def intensity_diff(row_res, col_res):
    difference_row = np.diff(row_res)
    difference_col = np.diff(col_res)
    difference_row = difference_row > 0
    difference_col = difference_col > 0
    return np.vstack((difference_row, difference_col)).flatten()
    #return difference_row
    #return np.vstack((difference_row, difference_col)) #str method

def file_hash(array):
    return md5(array).hexdigest()

def difference_score(image, height = 30, width = 30):
    gray = img_gray(image)
    row_res, col_res = resize(gray, height, width)
    difference = intensity_diff(row_res, col_res)
    
    return difference

def difference_score_dict_hash(image_list):
    ds_dict = {}
    duplicates = []
    hash_ds = []
    for image in image_list:
        ds = difference_score(image)
        hash_ds.append(ds)
        filehash = md5(ds).hexdigest()
        if filehash not in ds_dict:
            ds_dict[filehash] = image
        else:
            duplicates.append((image, ds_dict[filehash]) )
    
    return  duplicates, ds_dict, hash_ds

image_files = filter_images(image_files)
duplicates, ds_dict, hash_ds =difference_score_dict_hash(image_files)

print('duplicates', len(duplicates))


# for file_names in duplicates[:30]:
#     try:
#         plt.subplot(121),plt.imshow(imread(file_names[0]))
#         plt.title('Duplicate'), plt.xticks([]), plt.yticks([])

#         plt.subplot(122),plt.imshow(imread(file_names[1]))
#         plt.title('Original'), plt.xticks([]), plt.yticks([])
#         plt.show()
    
#     except OSError as e:
#         continue

#Now Let's try Hamming distance

def hamming_distance(image, image2):
    score =scipy.spatial.distance.hamming(image, image2)
    return score

#Hamming

def difference_score_dict(image_list):
    ds_dict = {}
    duplicates = []
    for image in image_list:
        ds = difference_score(image)
        
        if image not in ds_dict:
            ds_dict[image] = ds
        else:
            duplicates.append((image, ds_dict[image]) )
    
    return  duplicates, ds_dict

image_files = filter_images(image_files)
duplicates, ds_dict =difference_score_dict(image_files)

print('duplicates', len(duplicates))

print('ds_dict.keys()', ds_dict.keys())

import itertools
for k1,k2 in itertools.combinations(ds_dict, 2):
    if hamming_distance(ds_dict[k1], ds_dict[k2])< .05:
        duplicates.append((k1,k2))

print('duplicates', len(duplicates))

for file_names in duplicates[:31]:
    print('file_names', file_names)

# for file_names in duplicates[:31]:
#     try:
    
#         plt.subplot(121),plt.imshow(imread(file_names[0]))
#         plt.title('Duplicate'), plt.xticks([]), plt.yticks([])

#         plt.subplot(122),plt.imshow(imread(file_names[1]))
#         plt.title('Original'), plt.xticks([]), plt.yticks([])
#         plt.show()
    
#     except OSError as e:
#         continue

# os.remove()  #If we wanted to remove them`