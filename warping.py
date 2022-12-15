import cv2
import numpy as np
import sys
import threading
import resource
import os
import copy
from collections import deque

resource.setrlimit(resource.RLIMIT_STACK, (2**29,-1))
sys.setrecursionlimit(10**6)

def ReadImage(InputImagePath):
    Images = []                     # Input Images will be stored in this list.
    ImageNames = []                 # Names of input images will be stored in this list.
    
    # Checking if path is of file or folder.
    if os.path.isfile(InputImagePath):						    # If path is of file.
        InputImage = cv2.imread(InputImagePath)                 # Reading the image.
        
        # Checking if image is read.
        if InputImage is None:
            print("Image not read. Provide a correct path")
            exit()
        
        Images.append(InputImage)                               # Storing the image.
        ImageNames.append(os.path.basename(InputImagePath))     # Storing the image's name.

	# If path is of a folder contaning images.
    elif os.path.isdir(InputImagePath):
		# Getting all image's name present inside the folder.
        for ImageName in os.listdir(InputImagePath):
			# Reading images one by one.
            InputImage = cv2.imread(InputImagePath + "/" + ImageName)
			
            Images.append(InputImage)							# Storing images.
            ImageNames.append(ImageName)                        # Storing image's names.
        
    # If it is neither file nor folder(Invalid Path).
    else:
        print("\nEnter valid Image Path.\n")
        exit()

    return Images, ImageNames
        
Images, ImageNames = ReadImage('/home/sirabas/Documents/AGV/Segmentation/input_images')            # Reading all input images

for idx in range(len(Images)):
    img = Images[idx]
    orig_img = copy.deepcopy(Images[idx])

    IMAGE_H = 176
    IMAGE_W = 512

    src = np.float32([[20, IMAGE_H], [490, IMAGE_H], [100, 100], [400, 100]])
    dst = np.float32([[IMAGE_W/2 - 100, IMAGE_H], [IMAGE_W/2 + 100, IMAGE_H], [0, 0], [IMAGE_W,0]])
    M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
    Minv = cv2.getPerspectiveTransform(dst, src) # Inverse transformation

    img = img[30:(30+IMAGE_H), 0:IMAGE_W] # Apply np slicing for ROI crop
    warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H)) # Image warping


    cv2.namedWindow('original Image', cv2.WINDOW_NORMAL)
    cv2.namedWindow('warp', cv2.WINDOW_NORMAL)

    cv2.imshow('original Image', orig_img)
    warp = cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)
    cv2.imshow('warp', warp)

    nm = ImageNames[idx]
    newname = '/home/sirabas/Documents/AGV/Segmentation/outputwarp/'+nm[:-4] + 'warp' + '.png'
    print(newname)
    cv2.imwrite(newname, warp)