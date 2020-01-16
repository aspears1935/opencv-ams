
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;

/**
 * @function main
 */

#define CIRCLE 0
#define STRAIGHT 1

void MyPolygon( Mat img )
{
  int lineType = 8;

  int rows = img.rows;
  int cols = img.cols;

  /** Create some points */
  Point triangle_points[1][3];
  triangle_points[0][0] = Point( 0, rows-1 );
  triangle_points[0][1] = Point( 0, 0 );
  triangle_points[0][2] = Point( cols/2, rows-1 );
  
  const Point* ppt[1] = { triangle_points[0] };
  int npt[] = { 3 };

  fillPoly( img,
            ppt,
            npt,
            1,
            Scalar( 0, 0, 0 ),
            lineType );

  //Second triangle:
  triangle_points[0][0] = Point( cols/2, rows-1 );
  triangle_points[0][1] = Point( cols-1, 0 );
  triangle_points[0][2] = Point( cols-1, rows-1 );

  fillPoly( img,
            ppt,
            npt,
            1,
            Scalar( 0, 0, 0 ),
            lineType );
}

int main( int, char** argv )
{
  RNG rng;

  namedWindow("Sonar Map", WINDOW_AUTOSIZE);
  namedWindow("Sonar Noise", WINDOW_AUTOSIZE);
  namedWindow("Sonar Map Blur", WINDOW_AUTOSIZE);
  namedWindow("Sonar View", WINDOW_AUTOSIZE);
  namedWindow("Sonar Rot Rect", WINDOW_AUTOSIZE);

  Mat M, sonMap, sonNoise, sonMapBlur, rotated;
  sonMap = Mat::zeros(1000,1000, CV_8U);
  sonNoise = Mat::zeros(1000,1000,CV_8U);
  int circ_x, circ_y, circ_radius;

  float rectAngle = 0;
  Size rect_size(256,400);
  float roiTopLeft_x = 0;
  float roiTopLeft_y = 0;

  VideoWriter outputVideo;
  int fps=10;

  const string NAME = "output.avi";


  cout << "NAME=" << NAME << endl;
  cout << "cv::VideoWriter::fourcc('X','V','I','D')=" << cv::VideoWriter::fourcc('X','V','I','D') << endl;
  cout << "fps=" << fps << endl;
  cout << "S=" << Size(256,400) << endl;

  outputVideo.open(NAME, cv::VideoWriter::fourcc('X','V','I','D'), fps, Size(256,400),true); //CV_FOURCC('M','J','P','G')
  if (!outputVideo.isOpened())
    {
      cout  << "Could not open the output video for write: " << endl;
      return -1;
    }

  int i = 0;

  for(i=0; i<30; i++)
    {
      circ_x = rng.uniform(0,999);
      circ_y = rng.uniform(0,999);
      circ_radius = rng.uniform(4,10);
      circle(sonMap, Point(circ_x, circ_y), circ_radius, Scalar(120,120,120), -1);
    }

  for(i=0; i<300; i++)
    {
      circ_x = rng.uniform(0,999);
      circ_y = rng.uniform(0,999);
      circ_radius = rng.uniform(2,5);
      circle(sonMap, Point(circ_x, circ_y), circ_radius, Scalar(120,120,120), -1);
    }
  /*
    circle(sonMap, Point(500,500), 10, Scalar(150,150,150), -1);
    circle(sonMap, Point(500,500), 10, Scalar(150,150,150), -1);
    circle(sonMap, Point(550,550), 5, Scalar(150,150,150), -1);
    circle(sonMap, Point(450,520), 3, Scalar(150,150,150), -1);
    circle(sonMap, Point(350,30), 4, Scalar(150,150,150), -1);
    circle(sonMap, Point(250,750), 3, Scalar(150,150,150), -1);
    circle(sonMap, Point(150,200), 7, Scalar(150,150,150), -1);
    circle(sonMap, Point(100,100), 10, Scalar(150,150,150), -1);
    circle(sonMap, Point(40,70), 17, Scalar(150,150,150), -1);
    circle(sonMap, Point(100,700), 5, Scalar(150,150,150), -1);
    circle(sonMap, Point(400,500), 2, Scalar(150,150,150), -1);
    circle(sonMap, Point(800,300), 10, Scalar(150,150,150), -1);
    circle(sonMap, Point(900,20), 7, Scalar(150,150,150), -1);
  */

  if(CIRCLE)
    {
      for(i=0; i<361; i++)
	{
	  randn(sonNoise, 20, 100); //(img, mean, stddev)
	  
	  add(sonMap, sonNoise, sonMapBlur);
	  
	  GaussianBlur(sonMapBlur, sonMapBlur, Size(5,5), 0, 0);
	  GaussianBlur(sonMapBlur, sonMapBlur, Size(5,5), 0, 0);
	  GaussianBlur(sonMapBlur, sonMapBlur, Size(5,5), 0, 0);
	  
	  M = getRotationMatrix2D(Point(500,500), rectAngle, 1.0);
	  
	  warpAffine(sonMapBlur, rotated, M, Size(1000,1000), INTER_CUBIC);
	  
	  //      Mat roiRot(sonMapBlur, RotatedRect(Point2f(i*10,i*10),Size2f(256,799),rectAngle));
	  Mat roi(rotated, Rect(roiTopLeft_x+372, roiTopLeft_y+100, 256, 400));
	  
	  Mat output = roi.clone();
	  
	  MyPolygon(output);
	  
	  imshow("Sonar Map", sonMap);
	  imshow("Sonar Noise", sonNoise);
	  imshow("Sonar Map Blur", sonMapBlur);
	  imshow("Sonar Rot Rect", rotated);
	  imshow("Sonar View", output);
	  
	  rectAngle += 1;
	  
	  Mat colorOut;
	  cvtColor(output, colorOut, COLOR_GRAY2RGB);
	  
	  outputVideo << colorOut;
	  
	  waitKey(10);      
	}
    }
  
  if(STRAIGHT)
    {
      for(i=0; i<2400; i++)
	{
	  randn(sonNoise, 20, 100); //(img, mean, stddev)
	  
	  add(sonMap, sonNoise, sonMapBlur);
	  
	  GaussianBlur(sonMapBlur, sonMapBlur, Size(5,5), 0, 0);
	  GaussianBlur(sonMapBlur, sonMapBlur, Size(5,5), 0, 0);
	  GaussianBlur(sonMapBlur, sonMapBlur, Size(5,5), 0, 0);
	  
	  int add_x = 0;
	  int add_y = 0;

	  if(i<600)
	    {
	      add_x=i;
	      add_y=0;
	    }
	  else if(i < 1200)
	    {
	      add_x=599;
	      add_y=i-(600);
	    }
	  else if(i < 1800)
	    {
	      add_x=599-(i-1200);
	      add_y=599;
	    }
	  else
	    {
	      add_x=0;
	      add_y=599-(i-1800);
	    }

	  cout << "add_x=" << add_x << endl;
	  cout << "add_y=" << add_y << endl;

	  Mat roi(sonMapBlur, Rect(roiTopLeft_x+add_x, roiTopLeft_y+add_y, 256, 400));
	  
	  Mat output = roi.clone();
	  
	  MyPolygon(output);
	  
	  imshow("Sonar Map", sonMap);
	  imshow("Sonar Noise", sonNoise);
	  imshow("Sonar Map Blur", sonMapBlur);
	  imshow("Sonar View", output);
	  
	  Mat colorOut;
	  cvtColor(output, colorOut, COLOR_GRAY2RGB);
	  
	  outputVideo << colorOut;
	  
	  waitKey(10);      
	}
    }

  std::cout << "Finished Writing" << endl;
  
  return 0;
}
