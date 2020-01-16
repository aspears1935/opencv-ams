
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <cv.h>
#include "opencv2/opencv.hpp"
#include <cmath>

using namespace cv;
using namespace std;

RNG rng(12345);

int main(int argc, char** argv)
{

  //  Mat map = Mat::zeros(1440,200,CV_8U);
  //Mat map = Mat::zeros(800,1200,CV_8U);
  //Mat map = Mat::zeros(800,1200,CV_8U);

  //Mat hue1(Size(1000,200),CV_8UC3,Scalar(80,0,128));

  Mat hue720 = Mat::zeros(200,720,CV_8UC3);
  //  Mat roi(hue1,Rect(0,0,200,200));//x,y,width,height

  //roi=roi+100;


  /*    for(int i=0; i<180; i++)
    {
      Mat roi(hue1,Rect(4*i*(255/180),0,4*(255/180),200));//x,y,width,height
      roi=roi+Scalar(i,255,255);
      }*/
  

  for(int i=0; i<180; i++)
    {
      Mat roi(hue720,Rect(4*i,0,4,200));//x,y,width,height
      roi=roi+Scalar(i,255,255);
    }

  namedWindow("SRC", WINDOW_AUTOSIZE);

  //  vector<Mat> channels;
  //channels[0] = Mat::zeros(800,1200,CV_8U);
  //channels[1] = Mat::zeros(800,1200,CV_8U);
  //channels[2] = Mat::zeros(800,1200,CV_8U);


  
  //split(hsv_frame, channels);

  Mat hue512;
  resize(hue720,hue512,Size(512,200));
 
  Mat bgr;
  cvtColor(hue720, bgr, COLOR_HSV2BGR);
  imshow("SRC", bgr); //Good for colors
  waitKey(0);

  //  cvtColor(hue720, bgr, COLOR_HSV2BGR);
  //imshow("SRC", bgr); //Good for colors
  //waitKey(0);
  
  imwrite("hue720x200.png",bgr);

  return 0;



}
