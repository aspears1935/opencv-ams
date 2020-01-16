
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

/**
 * @function main
 */

#define CIRCLE 0
#define STRAIGHT 0
#define sonWidth 400 //256
#define sonHeight 600
#define mapWidth 3000//2000
#define mapHeight 2000
#define fps 5
#define PI 3.1415926535

bool CUSTOM = false; 

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

int main( int argc, char** argv )
{
  RNG rng;

  string imageFile0 = "cloudImgs/cloud1.png";
  string imageFile1 = "cloudImgs/cloud2.png";
  string imageFile2 = "cloudImgs/cloud3.png";
  string imageFile3 = "cloudImgs/cloud4.png";
  string imageFile4 = "cloudImgs/stucci1.jpg";
  string imageFile5 = "cloudImgs/stucci2.jpg";
  string imageFiles[6] = {imageFile0, imageFile1, imageFile2, imageFile3, imageFile4, imageFile5};

  int imageFileNum = 5; //Used to be option - now hardcoded.
  if((imageFileNum < 0) || (imageFileNum > 5))
    imageFileNum=5;

  char inputFileName[256];
  if(argc < 2)
    {
      cout << "Usage: ./sonarsim <input.csv>" << endl;
      return -1;
    }
  else
    {
      CUSTOM = true; 
      strcpy(inputFileName, argv[1]);
    }

  ifstream inFile(inputFileName);
  string tmpstring;


  //Get number of rows for each file input:
  getline(inFile,tmpstring,';'); //First line should just have length listed
  if(strcmp(tmpstring.c_str(),"length=")==0) //Found correct string
    cout << "Found length" << endl;
  else
    {
      cout << "Couldn't find length - " << tmpstring << endl;
      return 0;
    }
  getline(inFile,tmpstring,';'); //First line should just have length listed
  int lengthFile = (int)(atoi(tmpstring.c_str()));
  cout << "lengthFile=" << lengthFile << endl;
  getline(inFile,tmpstring,'\n'); //Discard rest of line

  getline(inFile,tmpstring,';');
  if(strcmp(tmpstring.c_str(),"x")==0)
    cout << "Found x heading" << endl;
  else
    {
      cout << "Couldn't find x heading" << endl;
      return 0;
    }

  getline(inFile,tmpstring,';');
  if(strcmp(tmpstring.c_str(),"y")==0)
    cout << "Found y heading" << endl;
  else
    {
      cout << "Couldn't find y heading" << endl;
      return 0;
    }

  getline(inFile,tmpstring,';');
  if(strcmp(tmpstring.c_str(),"yaw")==0)
    cout << "Found yaw heading" << endl;
  else
    {
      cout << "Couldn't find yaw heading" << endl;
      return 0;
    }

  float * x_arr;
  float * y_arr;
  float * yaw_arr;
  x_arr = new float[lengthFile];
  y_arr = new float[lengthFile];
  yaw_arr = new float[lengthFile];

  for(int i=0; i<lengthFile; i++)
    {
      getline(inFile,tmpstring,';');
      x_arr[i] = (float)(atof(tmpstring.c_str()));
      getline(inFile,tmpstring,';');
      y_arr[i] = (float)(atof(tmpstring.c_str()));
      getline(inFile,tmpstring,';');
      yaw_arr[i] = (float)(atof(tmpstring.c_str()));
      cout << "infile x,y,yaw=" << x_arr[i] << "," << y_arr[i] << "," << yaw_arr[i] << endl;
    }

  namedWindow("Sonar Map", WINDOW_AUTOSIZE);
  namedWindow("Sonar Noise", WINDOW_AUTOSIZE);
  namedWindow("Sonar Map Blur", WINDOW_AUTOSIZE);
  namedWindow("Sonar View", WINDOW_AUTOSIZE);
  namedWindow("Sonar Rot Rect", WINDOW_AUTOSIZE);
  namedWindow("ROI On Map", WINDOW_AUTOSIZE);

  Mat M, input, quadInput, sonMap, sonNoise, sonMapBlur, rotated;
  input = imread(imageFiles[imageFileNum],IMREAD_GRAYSCALE);
  cout << "Input Texture Image: Height=" << input.rows << " Width=" << input.cols << endl;
  quadInput = Mat::zeros(input.rows*4,input.cols*4,CV_8U);
  sonMap = Mat::zeros(mapHeight,mapWidth,CV_8U);
  sonNoise = Mat::zeros(sonMap.rows,sonMap.cols,CV_8U);
  Mat mapRoi1(quadInput, Rect(0, 0, input.cols, input.rows));
  input.copyTo(mapRoi1);
  Mat mapRoi2(quadInput, Rect(input.cols, 0, input.cols, input.rows));
  input.copyTo(mapRoi2);
  Mat mapRoi3(quadInput, Rect(0, input.rows, input.cols, input.rows));
  input.copyTo(mapRoi3);
  Mat mapRoi4(quadInput, Rect(input.cols, input.rows, input.cols, input.rows));
  input.copyTo(mapRoi4);
  Mat mapRoi5(quadInput, Rect(input.cols*2, 0, input.cols, input.rows));
  input.copyTo(mapRoi5);
  Mat mapRoi6(quadInput, Rect(input.cols*2, input.rows, input.cols, input.rows));
  input.copyTo(mapRoi6);
  Mat mapRoi7(quadInput, Rect(input.cols*3, 0, input.cols, input.rows));
  input.copyTo(mapRoi7);
  Mat mapRoi8(quadInput, Rect(input.cols*3, input.rows, input.cols, input.rows));
  input.copyTo(mapRoi8);  
  
  Mat mapRoi9(quadInput, Rect(0,0,input.cols*4,input.rows*2));
  Mat mapRoi10(quadInput, Rect(0,input.rows*2,input.cols*4,input.rows*2));
  mapRoi9.copyTo(mapRoi10);

  Mat mapRoi(quadInput, Rect(0,0, mapWidth, mapHeight));
  mapRoi.copyTo(sonMap);

  imwrite("sonMap.jpg",sonMap);

  int circ_x, circ_y, circ_radius;

  float rectAngle = 0;
  Size rect_size(sonWidth,sonHeight);
  float roiTopLeft_x = 0;
  float roiTopLeft_y = 0;

  VideoWriter outputVideo;

  const string NAME = "output.avi";


  cout << "NAME=" << NAME << endl;
  cout << "cv::VideoWriter::fourcc('X','V','I','D')=" << cv::VideoWriter::fourcc('X','V','I','D') << endl;
  cout << "fps=" << fps << endl;
  cout << "S=" << Size(sonWidth,sonHeight) << endl;

  outputVideo.open(NAME, cv::VideoWriter::fourcc('X','V','I','D'), fps, Size(sonWidth,sonHeight),true); //CV_FOURCC('M','J','P','G')
  if (!outputVideo.isOpened())
    {
      cout  << "Could not open the output video for write: " << endl;
      return -1;
    }

  int i = 0;

  /*  for(i=0; i<30; i++)
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
      }*/

      circle(sonMap, Point(mapWidth/2-800, mapHeight/2-800), 40, Scalar(20,20,20), -1);
      circle(sonMap, Point(mapWidth/2+200, mapHeight/2+200), 40, Scalar(20,20,20), -1);

      circle(sonMap, Point(mapWidth/2-200, mapHeight/2-200), 7, Scalar(255,255,255), -1);
      circle(sonMap, Point(mapWidth/2-205, mapHeight/2-205), 7, Scalar(255,255,255), -1);
      circle(sonMap, Point(mapWidth/2-205, mapHeight/2-200), 7, Scalar(255,255,255), -1);

  imwrite("sonMap.jpg",sonMap);


  if(CIRCLE)
    {
      for(i=0; i<361; i++)
	{
	  randn(sonNoise, 0, 30); //(img, mean, stddev) was mean=20, stddev=100
	  
	  add(sonMap, sonNoise, sonMapBlur);
	  
	  //GaussianBlur(sonMapBlur, sonMapBlur, Size(5,5), 0, 0);
	  GaussianBlur(sonMapBlur, sonMapBlur, Size(5,5), 0, 0);
	  GaussianBlur(sonMapBlur, sonMapBlur, Size(5,5), 0, 0);
	  
	  M = getRotationMatrix2D(Point(mapWidth/2,mapHeight/2), rectAngle, 1.0);
	  
	  warpAffine(sonMapBlur, rotated, M, Size(mapWidth, mapHeight), INTER_CUBIC);
	  
	  //      Mat roiRot(sonMapBlur, RotatedRect(Point2f(i*10,i*10),Size2f(256,799),rectAngle));
	  Mat roi(rotated, Rect(roiTopLeft_x+(mapWidth/2)-(sonWidth/2), roiTopLeft_y+(mapHeight/2)-sonHeight, sonWidth, sonHeight));
	  
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
      int trajHeight = 100;
      int trajWidth = 200;
      for(i=0; i<2*(trajHeight+trajWidth); i++) //544,544,200,200
	{
	  randn(sonNoise, 0, 30); //(img, mean, stddev) was mean=20, stddev=100
	  
	  add(sonMap, sonNoise, sonMapBlur);
	  
	  //GaussianBlur(sonMapBlur, sonMapBlur, Size(5,5), 0, 0);
	  GaussianBlur(sonMapBlur, sonMapBlur, Size(5,5), 0, 0);
	  GaussianBlur(sonMapBlur, sonMapBlur, Size(5,5), 0, 0);
	  
	  int add_x = 0;
	  int add_y = 0;

	  /*	  if(i<544)
	    {
	      add_x=i;
	      add_y=0;
	    }
	  else if(i < 544+200)
	    {
	      add_x=543;
	      add_y=i-(544);
	    }
	  else if(i < 544+200+544)
	    {
	      add_x=543-(i-744);
	      add_y=199;
	    }
	  else
	    {
	      add_x=0;
	      add_y=199-(i-1288);
	    }
	  */

	  if(i<trajWidth)
	    {
	      add_x=i;
	      add_y=0;
	    }
	  else if(i < trajWidth+trajHeight)
	    {
	      add_x=trajWidth-1;
	      add_y=i-(trajWidth);
	    }
	  else if(i < trajWidth+trajHeight+trajWidth)
	    {
	      add_x=(trajWidth-1)-(i-(trajWidth + trajHeight));
	      add_y=trajHeight-1;
	    }
	  else
	    {
	      add_x=0;
	      add_y=(trajHeight-1)-(i-(trajWidth + trajHeight + trajWidth));
	    }

	  cout << "add_x,y=" << add_x << "," << add_y << endl;

	  Mat roi(sonMapBlur, Rect(roiTopLeft_x+add_x, roiTopLeft_y+add_y, sonWidth, sonHeight));
	  
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

  if(CUSTOM)
    {
      //      int trajHeight = 100;
      //      int trajWidth = 200;
      int add_x = x_arr[1]-x_arr[0];
      int add_y = y_arr[1]-y_arr[0];
      int add_yaw = yaw_arr[1]-yaw_arr[0];
      //roiTopLeft_x+(mapWidth/2)-(sonWidth/2); 
      //roiTopLeft_y+(mapHeight/2)-sonHeight;

      //Initial rotation:
      M = getRotationMatrix2D(Point(x_arr[0],y_arr[0]), yaw_arr[0], 1.0);
      warpAffine(sonMap, rotated, M, Size(mapWidth, mapHeight), INTER_CUBIC);

      for(i=0; i<lengthFile; i++) //544,544,200,200
	{
	  randn(sonNoise, 0, 30); //(img, mean, stddev) was mean=20, stddev=100
	  
	  add(sonMap, sonNoise, sonMapBlur);
	  
	  //GaussianBlur(sonMapBlur, sonMapBlur, Size(5,5), 0, 0);
	  GaussianBlur(sonMapBlur, sonMapBlur, Size(5,5), 0, 0);
	  GaussianBlur(sonMapBlur, sonMapBlur, Size(5,5), 0, 0);

	  M = getRotationMatrix2D(Point(x_arr[i],y_arr[i]), yaw_arr[i], 1.0);
	  warpAffine(sonMapBlur, rotated, M, Size(mapWidth, mapHeight), INTER_CUBIC);

	  //	  Mat roiOnMap = rotated.clone();//
	  Mat roiOnMap = Mat::zeros(mapHeight,mapWidth,CV_8U);
	  
	  /*	  add_x=x_arr[i];
	  add_y=y_arr[i];
	  add_yaw=yaw_arr[i];
	  */
	  cout << "x,y,yaw=" << x_arr[i] << "," << y_arr[i] << "," << yaw_arr[i] << endl;

	  Mat roi(rotated, Rect(x_arr[i]-sonWidth/2, y_arr[i]-sonHeight, sonWidth, sonHeight));
	  
	  //Draw location of rectangle
	  //	  rectangle(roiOnMap, Rect(x_arr[i]-sonWidth/2, y_arr[i]-sonHeight, sonWidth, sonHeight), Scalar(255,255,255), 8);
	  line(roiOnMap, Point(x_arr[i],y_arr[i]), Point(x_arr[i]+sonHeight*sin(yaw_arr[i]*PI/180),y_arr[i]-sonHeight*cos(yaw_arr[i]*PI/180)), Scalar(255,255,255),8);
	  line(roiOnMap, Point(x_arr[i]+sonHeight*sin(yaw_arr[i]*PI/180)-sonWidth*cos(yaw_arr[i]*PI/180)/2, y_arr[i]-sonHeight*cos(yaw_arr[i]*PI/180)-sonWidth*sin(yaw_arr[i]*PI/180)/2), Point(x_arr[i]+sonHeight*sin(yaw_arr[i]*PI/180)+sonWidth*cos(yaw_arr[i]*PI/180)/2, y_arr[i]-sonHeight*cos(yaw_arr[i]*PI/180)+sonWidth*sin(yaw_arr[i]*PI/180)/2), Scalar(255,255,255),8);

	  Mat output = roi.clone();
	  
	  MyPolygon(output);
	  
	  imshow("Sonar Map", sonMap);
	  imshow("Sonar Noise", sonNoise);
	  imshow("Sonar Map Blur", sonMapBlur);
	  imshow("Sonar View", output);
	  imshow("Sonar Rot Rect", rotated);
	  imshow("ROI On Map", roiOnMap);

	  Mat colorOut;
	  cvtColor(output, colorOut, COLOR_GRAY2RGB);
	  
	  outputVideo << colorOut;
	  
	  waitKey(10);      
	}
    }

  std::cout << "Finished Writing" << endl;
  
  return 0;
}
