import os
from unittest import result
import cv2
import math
import numpy as np
import threading
import resource
import sys
import copy 
from collections import deque

resource.setrlimit(resource.RLIMIT_STACK, (2**29,-1))
sys.setrecursionlimit(10**6)

# Threshold by which lines will be rejected wrt the horizontal
REJECT_DEGREE_TH = 10.0


def ReadImage(InputImagePath):
    Images = []                    
    ImageNames = []                
    
    # Checking if path is of file or folder.
    if os.path.isfile(InputImagePath):						    # If path is of file.
        InputImage = cv2.imread(InputImagePath)                
        
        # Checking if image is read.
        if InputImage is None:
            print("Image not read. Provide a correct path")
            exit()
        
        Images.append(InputImage)                               
        ImageNames.append(os.path.basename(InputImagePath))     
	# If path is of a folder contaning images.
    elif os.path.isdir(InputImagePath):
		# Getting all image's name present inside the folder.
        for ImageName in sorted(os.listdir(InputImagePath)):
			# Reading images one by one.
            InputImage = cv2.imread(InputImagePath + "/" + ImageName)
			
            Images.append(InputImage)							
            ImageNames.append(ImageName)                        
        
    # If it is neither file nor folder(Invalid Path).
    else:
        print("\nEnter valid Image Path.\n")
        exit()

    return Images, ImageNames
        
Images, ImageNames = ReadImage('/home/sirabas/Documents/AGV/Segmentation/input_images')            # Reading all input images
SegImages, SegImageNames = ReadImage('/home/sirabas/Documents/AGV/Segmentation/segmented_images')

def FilterLines(Lines):
    FinalLines = []
    
    for Line in Lines:
        [[x1, y1, x2, y2]] = Line

        # Calculating equation of the line: y = mx + c
        if x1 != x2:
            m = (y2 - y1) / (x2 - x1)
        else:
            m = 100000000
        c = y2 - m*x2
        # theta will contain values between -90 -> +90. 
        theta = math.degrees(math.atan(m))

        # Rejecting lines of slope near to 0 degree or 90 degree and storing others
        if REJECT_DEGREE_TH <= abs(theta) <= (90 - REJECT_DEGREE_TH):
            l = math.sqrt( (y2 - y1)**2 + (x2 - x1)**2 )    # length of the line
            FinalLines.append([x1, y1, x2, y2, m, c, l])

    
    # Removing extra lines 
    # (we might get many lines, so we are going to take only longest 15 lines 
    # for further computation because more than this number of lines will only 
    # contribute towards slowing down of our algo.)
    FinalLines = sorted(FinalLines, key=lambda x: x[-1], reverse=True)
    if len(FinalLines) > 15:
        FinalLines = FinalLines[:15]
    
    return FinalLines
    


def GetLines(Image):
    # Converting to grayscale
    GrayImage = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
    # Blurring image to reduce noise.
    BlurGrayImage = cv2.GaussianBlur(GrayImage, (5, 5), 1)
    # Generating Edge image
    EdgeImage = cv2.Canny(BlurGrayImage, 40, 130)
    
    mask = np.zeros(EdgeImage.shape[:2], dtype = np.uint8)
    cv2.rectangle(mask , (0, int(0.3*EdgeImage.shape[0])), (EdgeImage.shape[1] ,EdgeImage.shape[0]), 255, -1)
    MaskedImage = cv2.bitwise_and(EdgeImage, mask)

    # Finding Lines in the image
    Lines = cv2.HoughLinesP(MaskedImage, 2, np.pi / 180, 40, 40, 15)
    
    # Check if lines found and exit if not.
    Image = [255,255,255]
    if Lines is None:
        print("Not enough lines found in the image for Vanishing Point detection.")
        exit(0)
    
    # Filtering Lines wrt angle
    FilteredLines = FilterLines(Lines)

    return FilteredLines
    

def GetVanishingPoint(Lines):
    # We will apply RANSAC inspired algorithm for this. We will take combination 
    # of 2 lines one by one, find their intersection point, and calculate the 
    # total error(loss) of that point. Error of the point means root of sum of 
    # squares of distance of that point from each line.
    VanishingPoint = None
    MinError = 100000000000

    for i in range(len(Lines)):
        for j in range(i+1, len(Lines)):
            m1, c1 = Lines[i][4], Lines[i][5]
            m2, c2 = Lines[j][4], Lines[j][5]

            if m1 != m2:
                x0 = (c1 - c2) / (m2 - m1)
                y0 = m1 * x0 + c1

                err = 0
                for k in range(len(Lines)):
                    m, c = Lines[k][4], Lines[k][5]
                    m_ = (-1 / m)
                    c_ = y0 - m_ * x0

                    x_ = (c - c_) / (m_ - m)
                    y_ = m_ * x_ + c_

                    l = math.sqrt((y_ - y0)**2 + (x_ - x0)**2)

                    err += l**2

                err = math.sqrt(err)

                if MinError > err:
                    MinError = err
                    VanishingPoint = [x0, y0]
                
    return VanishingPoint

def area(x1, y1, x2, y2, x3, y3):
 
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1)
                + x3 * (y1 - y2)) / 2.0)
 
 
# A function to check whether point P(x, y)
# lies inside the triangle formed by
# A(x1, y1), B(x2, y2) and C(x3, y3)
def isInside(x1, y1, x2, y2, x3, y3, x, y):
 
    # Calculate area of triangle ABC
    A = area (x1, y1, x2, y2, x3, y3)
 
    # Calculate area of triangle PBC
    A1 = area (x, y, x2, y2, x3, y3)
     
    # Calculate area of triangle PAC
    A2 = area (x1, y1, x, y, x3, y3)
     
    # Calculate area of triangle PAB
    A3 = area (x1, y1, x2, y2, x, y)
     
    # Check if sum of A1, A2 and A3
    # is same as A
    if(A == A1 + A2 + A3):
        return True
    else:
        return False
 

iou = np.zeros(len(Images)+1)

for idx in range(len(Images)):
    print(idx)
    print(ImageNames[idx])
    print(SegImageNames[idx])

    Image = Images[idx]
    resultimage = copy.deepcopy(Images[idx])
    # Getting the lines form the image
    Lines = GetLines(Image)
    # Get vanishing point
    VanishingPoint = GetVanishingPoint(Lines)
    # Checking if vanishing point found
    if VanishingPoint is None:
        print("Vanishing Point not found. Possible reason is that not enough lines are found in the image for determination of vanishing point.")
        continue
    # Drawing lines and vanishing point
    for Line in Lines:
        cv2.line(Image, (Line[0], Line[1]), (Line[2], Line[3]), (255, 0, 0), 2)
    cv2.circle(Image, (int(VanishingPoint[0]), int(VanishingPoint[1])), 10, (0, 0, 255), -1)

    Line1 = Lines[0]
    Line2 = Lines[1]
    for i,Line in enumerate(Lines[1:]):
        if( abs( math.degrees(math.atan(Line1[4])) - math.degrees(math.atan(Line[4])) ) > 30 ):
            Line2 = Line
            break
        if(i == len(Lines) - 2):
            Line2 = [int(VanishingPoint[0]), int(VanishingPoint[1]), 330, 220 ,0 ,0, 0]

    points = [[VanishingPoint[0], VanishingPoint[1]]]
    if(Line1[1] < Line1[3]):
        cv2.line(Image , (Line1[2], Line1[3]), (int(VanishingPoint[0]), int(VanishingPoint[1])) , (0,255,5), 2)
        points.append([Line1[2], Line1[3]])
    else:
        cv2.line(Image , (Line1[0], Line1[1]), (int(VanishingPoint[0]), int(VanishingPoint[1])) , (0,255,5), 2)
        points.append([Line1[0], Line1[1]])
    if(Line2[1] < Line2[3]):
        cv2.line(Image , (Line2[2], Line2[3]), (int(VanishingPoint[0]), int(VanishingPoint[1])) , (0,255,5), 2)
        points.append([Line2[2], Line2[3]])
    else:
        cv2.line(Image , (Line2[0], Line2[1]), (int(VanishingPoint[0]), int(VanishingPoint[1])) , (0,255,5), 2)
        points.append([Line2[0], Line2[1]])
    sumb = 0
    sumg = 0
    sumr = 0

    insidepoints = []
    for i in range(Image.shape[0]):
        for j in range(Image.shape[1]):
            if (isInside(points[0][0], points[0][1], points[1][0], points[1][1], points[2][0], points[2][1], j, i)):
                Image[i,j] = [255,255,255]
                insidepoints.append([i,j])
                sumb += resultimage[i,j,0]
                sumg += resultimage[i,j,1]
                sumr += resultimage[i,j,2]
    meanb = sumb/len(insidepoints)
    meang = sumg/len(insidepoints)
    meanr = sumr/len(insidepoints)
    stdb = np.std([resultimage[i,j,0] for [i,j] in insidepoints])
    stdg = np.std([resultimage[i,j,1] for [i,j] in insidepoints])
    stdr = np.std([resultimage[i,j,2] for [i,j] in insidepoints])

    midpointx = int((points[1][0] + points[2][0])/2 )
    midpointy = int((points[1][1] + points[2][1])/2 )
    seed1 = [ midpointx , midpointy ]
    seed2 = [ int((midpointx + points[0][0])/2), int((midpointy + points[0][1])/2) ]
    cv2.circle(Image, (seed1[0], seed1[1]), 10, (0, 0, 255), -1)
    cv2.circle(Image, (seed2[0], seed2[1]), 10, (0, 0, 255), -1)

    a = 1
    while True :
        j = seed1[0]
        i = seed1[1]
        if not (((resultimage[i, j , 0] >= (meanb - a*stdb)) and (resultimage[i, j , 0] <= (meanb + a*stdb))) and ((resultimage[i, j , 1] >= (meang - a*stdg)) and (resultimage[i, j , 1] <= (meang + a*stdg))) and ((resultimage[i, j , 2] >= (meanr - a*stdr)) and (resultimage[i, j , 2] <= (meanr + a*stdr)))):
            seed1[0] -= 10
            seed1[1] -= 10
        else:
            break
    
    while True :
        j = seed2[0]
        i = seed2[1]
        if not (((resultimage[i, j , 0] >= (meanb - a*stdb)) and (resultimage[i, j , 0] <= (meanb + a*stdb))) and ((resultimage[i, j , 1] >= (meang - a*stdg)) and (resultimage[i, j , 1] <= (meang + a*stdg))) and ((resultimage[i, j , 2] >= (meanr - a*stdr)) and (resultimage[i, j , 2] <= (meanr + a*stdr)))):
            seed2[0] -= 10
            seed2[1] -= 10
        else:
            break


    a = 1.6

    def floodfill2(i,j):
        stack = deque()
        stack.append([i,j])
        while(len(stack) != 0):
            [i,j] = stack.pop()
            if ((i in range(resultimage.shape[0])) and (j in range(resultimage.shape[1]))):
                if not (resultimage[i,j] == [255,255,255]).all():
                    if(((resultimage[i, j , 0] >= (meanb - a*stdb)) and (resultimage[i, j , 0] <= (meanb + a*stdb))) and ((resultimage[i, j , 1] >= (meang - a*stdg)) and (resultimage[i, j , 1] <= (meang + a*stdg))) and ((resultimage[i, j , 2] >= (meanr - a*stdr)) and (resultimage[i, j , 2] <= (meanr + a*stdr)))):
                        resultimage[i, j] = [255,255,255]
                        stack.append([i+1, j])
                        stack.append([i, j+1])
                        stack.append([i-1, j])
                        stack.append([i, j-1])
                    else:
                        resultimage[i, j] = [0,0,0]



    th1 = threading.Thread(target = floodfill2,args = (seed1[1] , seed1[0]))
    th1.start()
    th2 = threading.Thread(target = floodfill2,args = (seed2[1] , seed2[0]))
    th2.start()
    th1.join()
    th2.join()

    #Showing the final image
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow('Image', Image)
    cv2.namedWindow("floodedImage", cv2.WINDOW_NORMAL)
    cv2.imshow('floodedImage', resultimage)
    cv2.namedWindow("Ground Truth", cv2.WINDOW_NORMAL)
    cv2.imshow('Ground Truth', SegImages[idx])
    
    bw = copy.deepcopy(resultimage)

    for i in range(bw.shape[0]):
        for j in range(bw.shape[1]):
            if (bw[i][j] != [255,255,255]).all():
                bw[i][j] = [0,0,0]

    #bw = cv2.dilate(bw, (5,5))

    intersection = np.logical_and(bw, SegImages[idx])
    union = np.logical_or(bw, SegImages[idx])
    iou_score = np.sum(intersection) / np.sum(union)
    iou[idx] = iou_score

    cv2.namedWindow("My segmentation", cv2.WINDOW_NORMAL)
    cv2.imshow('My segmentation',  bw)
    nm = ImageNames[idx]
    newname = '/home/sirabas/Documents/AGV/Segmentation/output images/'+nm[:-4] + 'segmented' + '.png'
    print(newname)
    #cv2.imwrite(newname, bw)
    cv2.waitKey(0)

sumiou = 0
count = 0
for i in range(len(Images)):
    if(iou[i] > 0):
        sumiou += iou[i]
        count += 1

meaniou = sumiou/count
iou[-1] = np.mean(iou[:-1])
#np.savetxt("iou values.csv", iou)
print("Mean IOU = ", np.mean(iou[:-1]) )



