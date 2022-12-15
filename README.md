# Drivable_Area_Segmentation

Method used to accomplish drivable area segmentation:
1. Prediction of Vanishing point using a voting system of lines detected using HoughLines (Scope for improvement using Textural gradient - Gabor Filters)
2. Determining the Region of Interest as a triangle using Vanishing point and two of the best fit lines.
3. Sample two points from the Region of interest - 1.mid-point between two other vetices of triangle 2.mid-point of the point 1 and vanishing point 
4. Divide the ROI into two halves. Find the mean and std of pixel values of two regions.
5. Use the two points as seed points with respective ROI statistics to perform flood-fill.

We can perform Perspective Transform to get the bird's eye view. For which the code and output are given.
The mean IOU values were calculated between the GT and our prediction, which are given in IOU_values.csv
This project was an attempt to perform Drivable area segmenentation with basic IP methods, rather then computationally expensive Machine Learning methods. 
There is a lot of scope for improvement. The clouds are also detected because I have just taken pixels with value 255 to be considered. We can go about this by storing the predicted pixels separately rather than making them white in the original image.

# Example
Input Image

![monchengladbach_000000_009615_leftImg8bit](https://user-images.githubusercontent.com/106699115/207909968-feaefa0a-b398-4b33-80f4-3939b325a415.png)

Estimated Output

![monchengladbach_000000_009615_leftImg8bitsegmented](https://user-images.githubusercontent.com/106699115/207910011-3791042f-3eb7-4eef-a900-13d9d0f9795e.png)

Ground Truth

![monchengladbach_000000_009615_gtFine_color](https://user-images.githubusercontent.com/106699115/207910030-38f190f2-79f7-4840-b556-6b49a4ae11a4.png)
