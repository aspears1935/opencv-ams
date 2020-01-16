/**
 * @file CannyDetector_Demo.cpp
 * @brief Sample code showing how to detect edges using the Canny Detector
 * @author OpenCV team
 */

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <iostream> 
#include <cmath>

#define PI 3.14159265359

using namespace cv;
using namespace std;

Mat src, src_gray, dst;

int main( int, char** argv )
{
  /// Load an image
  src = imread( argv[1] );

  if( !src.data )
    { return -1; }
  

 

  /// Create a matrix of the same type and size as src (for dst)
  dst.create( src.size(), src.type() );

  /// Convert the image to grayscale
  cvtColor( src, src_gray, COLOR_BGR2GRAY );

  //DEBUG
  //float temp[5][5] = {{1, 2, 3, 4, 5},{6,7,8,9,10},{1,2,3,4,5},{1,2,3,4,5},{11,12,13,14,15}};
  //src_gray = Mat(5,5,CV_32F, temp);
   //END DEBUG

   //std::cout << "SRC GRAY:" << endl << src_gray << endl; 

  /// Create a window
  namedWindow("Input Image", WINDOW_AUTOSIZE );
  namedWindow("Histogram", WINDOW_NORMAL);

  Mat dx(src_gray.rows, src_gray.cols, CV_32F);
  Mat dy(src_gray.rows, src_gray.cols, CV_32F);
  int aperture_size = 3;

  Sobel(src_gray, dx, CV_32F, 1, 0, aperture_size, 1, 0, cv::BORDER_REPLICATE);
  Sobel(src_gray, dy, CV_32F, 0, 1, aperture_size, 1, 0, cv::BORDER_REPLICATE);

  //std::cout << "DX" << endl << dx << endl;
  //std::cout << "DY" << endl << dy << endl;

  Mat grad_mag(src_gray.rows, src_gray.cols, CV_32F);
  Mat grad_ang(src_gray.rows, src_gray.cols, CV_32F);

  float* dx_ptr = dx.ptr<float>(0);
  float* dy_ptr = dy.ptr<float>(0);
  float* mag_ptr = grad_mag.ptr<float>(0);
  float* ang_ptr = grad_ang.ptr<float>(0);

  //Calculate the gradient magnitudes and angles
  for(long i = 0; i<src_gray.rows*src_gray.cols; i++)
    {
      mag_ptr[i] = sqrt(dx_ptr[i]*dx_ptr[i] + dy_ptr[i]*dy_ptr[i]);
      ang_ptr[i] = atan(dy_ptr[i]/dx_ptr[i]);
    }

  //std::cout << "MAG:" << endl << grad_mag << endl;
  //std::cout << "ANG:" << endl << grad_ang << endl;

  vector<Mat> tempchannels;
  Mat grad_channels;
  tempchannels.push_back(grad_mag);
  tempchannels.push_back(grad_ang);
  merge(tempchannels, grad_channels); //Create 1 Mat with both mag and angle

  //std::cout << "MERGED:" << endl << grad_channels << endl;

  int ang_bins = 8;
  int mag_bins = 50;
  int histSize[] = {mag_bins, ang_bins};
  float mag_ranges[] = {0,50};
  float ang_ranges[] = {-PI,PI};
  const float* ranges[] = {mag_ranges, ang_ranges};
  int channels[] = {0,1};
  MatND histogram;
  calcHist(&grad_channels, 1, channels, Mat(), histogram, 2, histSize, ranges, true, false);
  normalize(histogram, histogram, 0, 1, NORM_MINMAX, -1, Mat());

  cout << histogram << endl;  
  imshow("Histogram", histogram);
  imshow("Input Image", src);

  /// Wait until user exit program by pressing a key
  waitKey(0);
  
  return 0;
}
