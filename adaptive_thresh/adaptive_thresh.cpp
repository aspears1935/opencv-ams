#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdlib.h>
#include <stdio.h>

using namespace cv;
using namespace std;

int blocksize = 3;
int mean_offset = 35;
int const max_blocksize = 25;
int const max_offset = 100;
int const max_BINARY_value = 255;

Mat src, dst;

char thresh_window[] = "Adaptive Threshold";

char blocksize_tbar[] = "Blocksize (2n+1)";
char offset_tbar[] = "Mean Offset - 50";

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
   createTrackbar( blocksize_tbar,
                  thresh_window, &blocksize,
                  max_blocksize, Threshold_Demo );

   createTrackbar( offset_tbar,
                  thresh_window, &mean_offset,
                  max_offset, Threshold_Demo );


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

  adaptiveThreshold( src, dst, max_BINARY_value, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 2*blocksize+3, mean_offset-50 );

  imshow( thresh_window, dst );
}
