/**
 * @file BasicLinearTransforms.cpp
 * @brief Simple program to change contrast and brightness
 * @author OpenCV team
 */

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdlib.h>
#include <stdio.h>

using namespace cv;
using namespace std;

int threshold_value = 0;
int threshold_type = 3;;
int const max_value = 255;
int const max_type = 4;
int const max_BINARY_value = 255;

Mat src, dst;

char thresh_window[] = "Threshold Demo";

char trackbar_type[] = "Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted";
char trackbar_value[] = "Value";

/// Function headers
void Threshold_Demo( int, void* );

/**
 * @function main
 * @brief Main function
 */
int main( int, char** argv )
{
   /// Read image given by user
   src = imread( argv[1] );
   dst = Mat::zeros( src.size(), src.type() );

   /// Create Windows
   namedWindow("Original Image", CV_WINDOW_AUTOSIZE);
   namedWindow("New Image", CV_WINDOW_AUTOSIZE);
   namedWindow(thresh_window, CV_WINDOW_AUTOSIZE);

   // Create Trackbars
   createTrackbar( trackbar_type,
                  thresh_window, &threshold_type,
                  max_type, Threshold_Demo );

   createTrackbar( trackbar_value,
                  thresh_window, &threshold_value,
                  max_value, Threshold_Demo );


   //Convert to Grayscale
   cvtColor(src, src, COLOR_BGR2GRAY);

   //Apply Histogram Equalization
   equalizeHist(src, src);

   //Call the function to initialize
   Threshold_Demo(0,0);

   /// Show stuff
   imshow("Original Image", src);
   imshow("New Image", dst);


   /// Wait until user press some key
   waitKey();
   
   //Save output image
   string outFileName = "dst.png";
   imwrite(outFileName, dst);
   
   return 0;
}


/**
 * @function Threshold_Demo
 */
void Threshold_Demo( int, void* )
{
  /* 0: Binary
     1: Binary Inverted
     2: Threshold Truncated
     3: Threshold to Zero
     4: Threshold to Zero Inverted
   */

  threshold( src, dst, threshold_value, max_BINARY_value,threshold_type );

  imshow( thresh_window, dst );
}
