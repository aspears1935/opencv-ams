
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <iomanip>

#include <cv.h>
#include "opencv2/opencv.hpp"
#include <cmath>

#define MIN_BOX_RADIUS 20 //was 8
#define MAP_SCALAR 10
#define IGNORE_HUE0 true
#define IGNORE_HUE255 false
#define MORPH_OPEN false
#define MORPH_CLOSE false //intrdoces MANY more bad features
#define USE_STATIC_THRESHOLDS false
#define VERBOSE true
#define STOP_BETWEEN_FRAMES false
#define IMAGE_NOTVIDEO true
#define THRESH_GREEN_MORE false
#define WRITE_VIDEO false

#define BLUE_HIGH 151
#define BLUE_LOW 81
#define GREEN_HIGH 82
#define GREEN_LOW 60
#define MAX_COUNT 0.1
#define MIN_COUNT 0.0001
#define ZERO 0.00001
#define MIN_CONTOUR_AREA_BLUE 0.00006


using namespace cv;
using namespace std;

RNG rng(12345);

int main(int argc, char** argv)
{
  Mat input;
  //input = imread("input.png",IMREAD_COLOR);

  if(argc < 2)
    cout << "Not enough args" << endl << "Usage: ./hsv_split <video_file>" << endl;

  ofstream outfile("output.csv");
  outfile << "frame;numNotBlue;%NotBlue;objDetected;objDetectedBlue" << endl;

  Mat colormap = Mat::zeros(800,1200,CV_8UC3);
  namedWindow("COLOR MAP", WINDOW_AUTOSIZE);

  VideoCapture cap;
  Mat frame, h_thresh, h_morph, h_components;
  double frames;
  int height, width;
  bool objDetected = false;
  bool objDetectedBlue = false;
  bool objDetectedLocalMax = false;

  if(IMAGE_NOTVIDEO) //Input is an image, not a video
    {
      frame = imread(argv[1], IMREAD_COLOR);
      frames = 1;
      height = frame.rows;
      width = frame.cols;
    }
  else //Input is a video
    {
      cap = VideoCapture(argv[1]);
      if(!cap.isOpened())
	{
	  cout << "CANNOT OPEN INPUT VIDEO" << endl;
	  return -1;
	}
      double framePos = cap.get(CAP_PROP_POS_FRAMES); 
      double fourcc = cap.get(CAP_PROP_FOURCC);
      frames = cap.get(CAP_PROP_FRAME_COUNT);
      height = cap.get(CAP_PROP_FRAME_HEIGHT);
      width = cap.get(CAP_PROP_FRAME_WIDTH);

      cout << "FramePos=" << framePos << endl;

      cout << "Fourcc = " << fourcc << endl;
    }

      cout << "Frames = " << frames << endl;
      cout << "Height,Width = " << height << "," << width << endl;

      //Constants for mapping image data to map
      double fx = 1050; //Blender=1050
      double fy = 1050; //Blender=1050
      double altitude = 20; //Blender - estimate 5meters to ice?
      double posx = 100;
      double posy = 100;
      double FOVX = 2*(180/CV_PI)*atan((width/2)/fx);
      cout << "FOVX=" << FOVX << endl;
      double FOVY = 2*(180/CV_PI)*atan((height/2)/fy);
      cout << "FOVY=" << FOVY << endl;

      Mat map = Mat::zeros(800,1200,CV_8UC3);
      Mat mapBlue = Mat::zeros(800,1200,CV_8UC3);
      Mat mapLocalMax = Mat::zeros(800,1200,CV_8UC3);
     
      //Initialize for output images
  string fileNameArray[10];
  char lastStr[256];
  char * pch;
  pch = strtok(argv[1],"./");
  int i1 = 0;
  while(pch != NULL)
    {
      string tempstr(pch);
      strcpy(lastStr,tempstr.c_str());
      fileNameArray[i1] = lastStr;
      i1++;
      pch = strtok(NULL, "./");
    }  
  stringstream oss;

  namedWindow("SRC", WINDOW_AUTOSIZE);
  namedWindow("H", WINDOW_AUTOSIZE);
  namedWindow("H_MORPH", WINDOW_AUTOSIZE);
  namedWindow("H_COMPONENTS", WINDOW_AUTOSIZE);
  namedWindow("S", WINDOW_AUTOSIZE);
  namedWindow("V", WINDOW_AUTOSIZE);
  namedWindow("MAP", WINDOW_AUTOSIZE);
  namedWindow("MAP BLUE", WINDOW_AUTOSIZE);
  namedWindow("MAP LOCAL MAX", WINDOW_AUTOSIZE);
  namedWindow("Components Blue", WINDOW_AUTOSIZE);

  //Init video writers
  VideoWriter outputVideoHue;
  VideoWriter outputVideoHueMaxAnomaly;
  VideoWriter outputVideoHueThreshAnomaly;
  int outputFPS = 10;
  if(WRITE_VIDEO)
    {
      outputVideoHue.open("outputHue.avi",cv::VideoWriter::fourcc('X','V','I','D'), outputFPS, Size(width*2,height*2),true);
      if(!outputVideoHue.isOpened())
        {
          cout << "Could not open the output video for write" << endl;
          return -1;
        }

      outputVideoHueMaxAnomaly.open("outputHueMaxAnomaly.avi",cv::VideoWriter::fourcc('X','V','I','D'), outputFPS, Size(width*2,height*2),true);
      if(!outputVideoHueMaxAnomaly.isOpened())
        {
          cout << "Could not open the output video for write" << endl;
          return -1;
        }

      outputVideoHueThreshAnomaly.open("outputHueThreshAnomaly.avi",cv::VideoWriter::fourcc('X','V','I','D'), outputFPS, Size(width*2,height*2),true);
      if(!outputVideoHueThreshAnomaly.isOpened())
        {
          cout << "Could not open the output video for write" << endl;
          return -1;
        }
    }

  Mat histSummed, histSummedGreen, componentsLocalMax, histBlue1, histBlue2, histThreshBlue;
  Mat dilateBlue, erodeBlue, componentsBlue;

  vector<Mat> hsv_planes;

  /******************Main Loop*************************************/

  int i = 0;
  for(i=0; i<frames; i++)
   {
     cout << i+1 << "/" << frames << endl;
     Mat hsv_frame;

     if(!IMAGE_NOTVIDEO)
       cap >> frame;

     cvtColor(frame, hsv_frame, COLOR_BGR2HSV);

      split(hsv_frame, hsv_planes);

      //Get averages: Use itself as a mask to eliminate zeros from average
      double h_avg = mean(hsv_planes[0],hsv_planes[0])[0];
      double s_avg = mean(hsv_planes[1],hsv_planes[0])[0];

      if(VERBOSE)
	{
	  cout << "Average Values: H=" << h_avg << " S=" << s_avg << endl;
	}
      int h_threshold_low;
      int h_threshold_high;
      int s_threshold_low;
      int s_threshold_high;

      if(USE_STATIC_THRESHOLDS)
	{
	  h_threshold_low = 80; //average was 100 for sim
	  h_threshold_high = 120;
	  s_threshold_low = 10;//average was 50 for sim
	  s_threshold_high = 180;
	}
      else
	{
	  h_threshold_low = h_avg-20; //was 20
	  h_threshold_high = h_avg+20;
	  s_threshold_low = s_avg-40;
	  s_threshold_high = s_avg+130;
	}
	  int max_val = 255;



      Mat h_thresh1, h_thresh1tmp, h_thresh2, h_thresh2tmp, s_thresh, s_thresh1, s_thresh2;
      //Get values higher than threshold_high:
      if(IGNORE_HUE255)
	{
	  threshold(hsv_planes[0], h_thresh1tmp, 254, max_val, THRESH_TOZERO);
	  threshold(h_thresh1tmp, h_thresh1, h_threshold_high, max_val, THRESH_BINARY);
	}
      else
      threshold(hsv_planes[0], h_thresh1, h_threshold_high, max_val, THRESH_BINARY);
      
      //Get values lower than threshold_low:
      if(IGNORE_HUE0)
	{
	  threshold(hsv_planes[0], h_thresh2tmp, h_threshold_low, max_val, THRESH_TOZERO_INV);     
	  threshold(h_thresh2tmp, h_thresh2, 0, max_val, THRESH_BINARY);
	}
      else
	threshold(hsv_planes[0], h_thresh2, h_threshold_low, max_val, THRESH_BINARY_INV);
      add(h_thresh1, h_thresh2, h_thresh);

      threshold(hsv_planes[1], s_thresh1, s_threshold_high, max_val, THRESH_BINARY);
      threshold(hsv_planes[1], s_thresh2, s_threshold_low, max_val, THRESH_BINARY_INV);
      add(s_thresh1, s_thresh2, s_thresh);

      //Morphology. Opening = erode, then dilate:
      Mat element = getStructuringElement(MORPH_RECT,Size(3,3),Point(1,1));
      if(MORPH_OPEN)
	{
	  int numErodeIters = 1;
	  int numDilateIters = 1;
	  erode(h_thresh, h_morph, element,Point(-1,-1),numErodeIters);
	  dilate(h_morph, h_morph, element,Point(-1,-1),numDilateIters);
	}
      else if(MORPH_CLOSE)
	{
	  int numErodeIters = 1;
	  int numDilateIters = 1;
	  dilate(h_thresh, h_morph, element,Point(-1,-1),numDilateIters);
	  erode(h_morph, h_morph, element,Point(-1,-1),numErodeIters);
	}
      else
	{
	  cout << "Threshold image cloned" << endl;
	  h_morph=h_thresh.clone();
	}
      //Components:
      //Components:                                                                
      vector<vector<Point> > contours;
      vector<Vec4i> hierarchy;
      Mat tmp1 = h_thresh.clone();
      findContours(tmp1,contours,hierarchy,CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE);
      vector<vector<Point> > contours_poly(contours.size());
      vector<Rect> boundRect(contours.size());
      vector<Point2f> center(contours.size());
      vector<float> radius(contours.size());
      cvtColor(h_morph, h_components, COLOR_GRAY2BGR);

      if(contours.size() > 0)
	objDetected = true;

      for(int count = 0; count < contours.size(); count++)
	{
	  approxPolyDP(Mat(contours[count]),contours_poly[count], 3, true);
	  boundRect[count] = boundingRect(Mat(contours_poly[count]));
	  minEnclosingCircle((Mat)contours_poly[count], center[count], radius[count]);
	  if(radius[count] < MIN_BOX_RADIUS)
	    continue;
	  Scalar color = Scalar(rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255));
	  drawContours(h_components, contours_poly, count, color, 1, 8, vector<Vec4i>(),0,Point());
	  rectangle(h_components, boundRect[count].tl(), boundRect[count].br(), color, 2, 8, 0);

	  //DRAW MAP:
	  double xmap = altitude*tan((center[count].x-(width/2))*(FOVX/width)*(CV_PI/180));
	  double ymap = altitude*tan((center[count].y-(height/2))*(FOVY/height)*(CV_PI/180));
	  /*	  if(VERBOSE)
	  {
	      cout << "x,y img=" << center[count].x << "," << center[count].y << endl;
	      cout << "x,y map=" << xmap << "," << ymap << endl;
	      }*/
	  circle(map, Point(posx+(xmap*MAP_SCALAR),posy+(ymap*MAP_SCALAR)), radius[count]/6, Scalar(255,0,0),-1);

	}

      //--------------------Blue Thresholding-----------------//

      //Mat histThreshBlue, histBlue1, histBlue2;//151,81     
      threshold(hsv_planes[0], histBlue1, BLUE_HIGH, 255, THRESH_BINARY);//High vals
      threshold(hsv_planes[0], histBlue2, BLUE_LOW, 255, THRESH_TOZERO_INV); //Rm 0 Vals, if src>=thresh, dst=0                                 
      threshold(histBlue2, histBlue2, 1, 255, THRESH_BINARY); //Rm Low Vals   
      histThreshBlue = histBlue1+histBlue2;
      //Calculate percent not blue:
      int sumNotBlue = (int)sum(histThreshBlue)[0]/255;
      double pctNotBlue = (double)sumNotBlue/(histThreshBlue.rows*histThreshBlue.cols);

      if(abs(pctNotBlue) < ZERO)
	pctNotBlue = 0;

      //Mat noblue_hist, noblue_histT;;
      //calcHist( &histThreshBlue, 1, 0, Mat(), noblue_hist, 1, &histSize, &histRange, uniform, accumulate );
      //transpose(noblue_hist, noblue_histT);
      //  cout << noblue_histT << endl; 
      
      dilate(histThreshBlue, dilateBlue, getStructuringElement(MORPH_RECT,Size(3,3),Point(1,1)), Point(-1,-1),4);                                                   
      erode(dilateBlue, erodeBlue, getStructuringElement(MORPH_RECT,Size(3,3),Point(1,1)), Point(-1,-1),4);                                                
      
      //------Blue Components------//     
      //First open image:                                         
      vector<vector<Point> > contoursBlue;
      vector<Vec4i> hierarchyBlue;
      tmp1 = erodeBlue.clone();
      findContours(tmp1,contoursBlue,hierarchyBlue,RETR_CCOMP,CHAIN_APPROX_SIMPLE);
      
      vector<vector<Point> > contours_polyBlue(contoursBlue.size());
      vector<Rect> boundRectBlue(contoursBlue.size());
      vector<Point2f> centerBlue(contoursBlue.size());
      vector<float> radiusBlue(contoursBlue.size());
      vector<double> areaBlue(contoursBlue.size());
      
      if(contoursBlue.size() > 0)
	objDetectedBlue = true;
      componentsBlue = Mat::zeros(frame.rows,frame.cols,CV_8UC3);
      for(int count = 0; count < contoursBlue.size(); count++)
	{
	  approxPolyDP(Mat(contoursBlue[count]),contours_polyBlue[count], 3, true);
	  boundRectBlue[count] = boundingRect(Mat(contours_polyBlue[count]));
	  minEnclosingCircle((Mat)contours_polyBlue[count], centerBlue[count], radiusBlue[count]);
	  areaBlue[count] = contourArea(contoursBlue[count]);

	  
	  if(areaBlue[count] < (int)(MIN_CONTOUR_AREA_BLUE*frame.rows*frame.cols))
	    continue;
	  if(VERBOSE)
	    cout << "Blue component size: " << areaBlue[count] << endl;
	  Scalar color = Scalar(rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255));
	  drawContours(componentsBlue, contours_polyBlue, count, color, FILLED, 8, vector<Vec4i>(),0,Point());
	  rectangle(componentsBlue, boundRectBlue[count].tl(), boundRectBlue[count].br(), color, 2, 8, 0);                                                              

	  //DRAW BLUE MAP:
	  double xmap = altitude*tan((centerBlue[count].x-(width/2))*(FOVX/width)*(CV_PI/180));
	  double ymap = altitude*tan((centerBlue[count].y-(height/2))*(FOVY/height)*(CV_PI/180));
	  circle(mapBlue, Point(posx+(xmap*MAP_SCALAR),posy+(ymap*MAP_SCALAR)), radiusBlue[count]/6, Scalar(255,0,0),-1);
	}

      //---------------------- Local Max Mappi2ng -----------------------//
      int histSize = 180;
      float range[] = {0,180};
      const float* histRange = {range};
      bool uniform=true; bool accumulate=false;
      Mat h_hist, h_histT;
      calcHist(&hsv_planes[0], 1, 0, Mat(), h_hist, 1, &histSize, &histRange, uniform, accumulate);
      transpose(h_hist, h_histT);
      int MIN = MIN_COUNT*(frame.cols*frame.rows);
      int MAX = MAX_COUNT*(frame.cols*frame.rows);
      Mat histImgNew;
      histSummed = Mat::zeros(frame.rows,frame.cols,CV_8U);
      histSummedGreen = Mat::zeros(frame.rows,frame.cols,CV_8U);
      
      //Add all local maximums                                                           
      for(int i2=1; i2<histSize-1; i2++) //Have to go from 1:size-1 to leave 1 buffer       
	{
	  if((h_histT.at<float>(i2-1)<h_histT.at<float>(i2))&&(h_histT.at<float>(i2)>h_histT.at<float>(i2+1))) //local max                 
	    {
	      if((h_histT.at<float>(i2) < MIN)||(h_histT.at<float>(i2) > MAX)) //Check if between thresholds
		{
		  continue; //If outside thresholds, don't calculate                     
		}
	      
	      if((i2>BLUE_LOW)&&(i2<BLUE_HIGH)) //Check if between thresholds was 95       
		{
		  continue; //If outside thresholds, don't calculate                     
		}

	      if((i2==10)||(i2==30)||(i2==60)||(i2==165)||(i2==170))
		{
		  continue; //If 165, don't show - this is a weird harmonic of blue?     
		}
	      
	      threshold(hsv_planes[0], histImgNew, i2+1, 255, THRESH_TOZERO_INV); //was i2-1                                                                                   
	      if(i2>1)
		threshold(histImgNew, histImgNew, i2-2, 255, THRESH_BINARY); //was i2-1    
	      else
		threshold(histImgNew, histImgNew, 1, 255, THRESH_BINARY); //was i2-1      

	      if(THRESH_GREEN_MORE)
		{
		  if((i2>GREEN_LOW)&&(i2<GREEN_HIGH)) //Greens, close to blue 
		    {
		      erode(histImgNew, histImgNew, getStructuringElement(MORPH_RECT,Size(3, 3),Point(1,1)), Point(-1,-1),3);
		      dilate(histImgNew, histImgNew, getStructuringElement(MORPH_RECT,Size(3 ,3),Point(1,1)), Point(-1,-1),3);
		      histSummedGreen += histImgNew;
		    }
		  else
		    histSummed += histImgNew;
		}
	      else
		histSummed += histImgNew;

	      if(IMAGE_NOTVIDEO)
		{
		  oss.str("");
		  oss << fileNameArray[i1-2] << "_maxThresh" << i2 << ".png";
		  imwrite(oss.str(),histSummed);
		}

	    }
	}

      //------LocalMax Components------//     
      //First open image:                                         
      vector<vector<Point> > contoursLocalMax;
      vector<Vec4i> hierarchyLocalMax;
      tmp1 = histSummed.clone();
      findContours(tmp1,contoursLocalMax,hierarchyLocalMax,RETR_CCOMP,CHAIN_APPROX_SIMPLE);
      
      vector<vector<Point> > contours_polyLocalMax(contoursLocalMax.size());
      vector<Rect> boundRectLocalMax(contoursLocalMax.size());
      vector<Point2f> centerLocalMax(contoursLocalMax.size());
      vector<float> radiusLocalMax(contoursLocalMax.size());
      vector<double> areaLocalMax(contoursLocalMax.size());
      
      if(contoursLocalMax.size() > 0)
	objDetectedLocalMax = true;
      componentsLocalMax = Mat::zeros(frame.rows,frame.cols,CV_8UC3);
      for(int count = 0; count < contoursLocalMax.size(); count++)
	{
	  approxPolyDP(Mat(contoursLocalMax[count]),contours_polyLocalMax[count], 3, true);
	  boundRectLocalMax[count] = boundingRect(Mat(contours_polyLocalMax[count]));
	  minEnclosingCircle((Mat)contours_polyLocalMax[count], centerLocalMax[count], radiusLocalMax[count]);
	  areaLocalMax[count] = contourArea(contoursLocalMax[count]);

	  
	  if(areaLocalMax[count] < (int)(MIN_CONTOUR_AREA_BLUE*frame.rows*frame.cols))
	    continue;
	  if(VERBOSE)
	    cout << "LocalMax component size: " << areaLocalMax[count] << endl;
	  Scalar color = Scalar(rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255));
	  drawContours(componentsLocalMax, contours_polyLocalMax, count, color, FILLED, 8, vector<Vec4i>(),0,Point());
	  rectangle(componentsLocalMax, boundRectLocalMax[count].tl(), boundRectLocalMax[count].br(), color, 2, 8, 0);                                                              

	  //DRAW BLUE MAP:
	  double xmap = altitude*tan((centerLocalMax[count].x-(width/2))*(FOVX/width)*(CV_PI/180));
	  double ymap = altitude*tan((centerLocalMax[count].y-(height/2))*(FOVY/height)*(CV_PI/180));
	  circle(mapLocalMax, Point(posx+(xmap*MAP_SCALAR),posy+(ymap*MAP_SCALAR)), radiusLocalMax[count]/6, Scalar(255,0,0),-1);
	}


      //-------------------------------------------------------------------------

      cout << "NotBlue: " << sumNotBlue << "/" << histThreshBlue.rows*histThreshBlue.cols << "=" << pctNotBlue << endl;
      outfile << fixed;
      outfile << setprecision(5);
      outfile << i << ";" << sumNotBlue << ";" << (double)sumNotBlue/(histThreshBlue.rows*histThreshBlue.cols) << ";" << objDetected << ";" << objDetectedBlue << ";" << objDetectedLocalMax << ";" << endl;
      objDetected = false;
      objDetectedBlue = false;
      objDetectedLocalMax = false;

      if(STOP_BETWEEN_FRAMES)
	waitKey(0);
      else
	waitKey(10);

      //Plot Vehicle Location: (x=100m, y=40m, z=20m)
      Scalar green(0,255,0);
      Scalar red(0,0,255);
      if(contours.size() > 0)
	{
	  circle(map, Point(posx, posy), 2, green);
	}
      else
	{
	  circle(map, Point(posx, posy), 2, red);
	}
      
      if(contoursBlue.size() > 0)
	{
	  circle(mapBlue, Point(posx, posy), 2, green);
	}
      else
	{
	  circle(mapBlue, Point(posx, posy), 2, red);
	}

      if(contoursLocalMax.size() > 0)
	{
	  circle(mapLocalMax, Point(posx, posy), 2, green);
	}
      else
	{
	  circle(mapLocalMax, Point(posx, posy), 2, red);
	}

      //Draw NotBlue Map: 
      double area = height*width;
      circle(colormap, Point(posx, posy), 2, Scalar(5*255*pctNotBlue, 5*255*pctNotBlue, 5*255*pctNotBlue),-1); //DEBUG: Delete the 2x. just for all ice too faded

      if(VERBOSE)
	cout << "posx,posy=" << posx << "," << posy << endl;
      
      imshow("SRC",frame);
      //imshow("H", channels[0]);
      imshow("H", h_thresh); //Good for colors
      imshow("H_MORPH", h_morph); //Good for colors
      imshow("H_COMPONENTS", h_components); //Good for colors
      //imshow("S", channels[1]);
      imshow("S", s_thresh); //Good for blacks/whites
      imshow("V", hsv_planes[2]);
      imshow("MAP", map);
      imshow("MAP BLUE", mapBlue);
      imshow("MAP LOCAL MAX", mapLocalMax);
      imshow("COLOR MAP", colormap);
      imshow("Components Blue", componentsBlue);
      
      //Write Videos:
      Mat outVideoImg = Mat::zeros(frame.rows*2,frame.cols*2,CV_8UC3);
      Mat tmp10, tmp2, tmp3, tmp4, tmp5;

      resize(frame, tmp10, Size(width,height));
      Mat mapRoi1(outVideoImg, Rect(0, 0, frame.cols, frame.rows));
      tmp10.copyTo(mapRoi1);

      cvtColor(hsv_planes[0],tmp5,COLOR_GRAY2BGR);   
      resize(tmp5, tmp2, Size(width,height));
      Mat mapRoi2(outVideoImg, Rect(frame.cols, 0, frame.cols, frame.rows));
      tmp2.copyTo(mapRoi2);

      cvtColor(h_thresh,tmp5,COLOR_GRAY2BGR);
      resize(tmp5, tmp3, Size(width,height));
      Mat mapRoi3(outVideoImg, Rect(0, frame.rows, frame.cols, frame.rows));
      tmp3.copyTo(mapRoi3);

      //cvtColor(colormap, tmp5,COLOR_GRAY2BGR);
      //      colormap = colormap*10;
      resize(colormap, tmp4, Size(width,height));
      Mat mapRoi4(outVideoImg, Rect(frame.cols, frame.rows, frame.cols, frame.rows));
      tmp4.copyTo(mapRoi4);

      if(WRITE_VIDEO)
        outputVideoHue << outVideoImg;

      //Write Anomaly Video:                      
      outVideoImg = Mat::zeros(frame.rows*2,frame.cols*2,CV_8UC3);

      resize(frame, tmp10, Size(width,height));
      //      Mat mapRoi1(outVideoImg, Rect(0, 0, src.cols, src.rows));   
      tmp10.copyTo(mapRoi1);

      cvtColor(h_thresh,tmp5,COLOR_GRAY2BGR);
      resize(tmp5, tmp2, Size(width,height));
      //      Mat mapRoi2(outVideoImg, Rect(src.cols, 0, src.cols, src.rows));   
      tmp2.copyTo(mapRoi2);

      resize(h_components, tmp3, Size(width,height));
      //      Mat mapRoi3(outVideoImg, Rect(0, src.rows, src.cols, src.rows));
      tmp3.copyTo(mapRoi3);

      //cvtColor(mapTextureBoth,tmp5,COLOR_GRAY2BGR);                        
      resize(mapLocalMax, tmp4, Size(width,height));
      //      Mat mapRoi4(outVideoImg, Rect(src.cols, src.rows, src.cols, src.rows));                                                                             
      tmp4.copyTo(mapRoi4);

      if(WRITE_VIDEO)
        outputVideoHueMaxAnomaly << outVideoImg;

      //Write Thresh Blue Anomaly Video:                                                         
      outVideoImg = Mat::zeros(frame.rows*2,frame.cols*2,CV_8UC3);

      resize(frame, tmp10, Size(width,height));
      //      Mat mapRoi1(outVideoImg, Rect(0, 0, src.cols, src.rows));     
      tmp10.copyTo(mapRoi1);

      cvtColor(h_thresh,tmp5,COLOR_GRAY2BGR);
      resize(tmp5, tmp2, Size(width,height));
      //      Mat mapRoi2(outVideoImg, Rect(src.cols, 0, src.cols, src.rows));  
      tmp2.copyTo(mapRoi2);

      resize(componentsBlue, tmp3, Size(width,height));
      //      Mat mapRoi3(outVideoImg, Rect(0, src.rows, src.cols, src.rows));
      tmp3.copyTo(mapRoi3);

      //cvtColor(mapTextureBoth,tmp5,COLOR_GRAY2BGR);     
      resize(mapBlue, tmp4, Size(width,height));
      //      Mat mapRoi4(outVideoImg, Rect(src.cols, src.rows, src.cols, src.rows));                                                                             
      tmp4.copyTo(mapRoi4);

      if(WRITE_VIDEO)
        outputVideoHueThreshAnomaly << outVideoImg;


      if(STOP_BETWEEN_FRAMES)
	waitKey(0);
      else
	waitKey(10);
      
      if(i < 200)
	posx=posx+(0.5*MAP_SCALAR);
      else if(i < 300)
	posy=posy+(0.4*MAP_SCALAR);
      else if(i < 500)
	posx=posx-(0.5*MAP_SCALAR);
      else //i > 500
	posy=posy-(0.4*MAP_SCALAR);


   }

  

  if(!IMAGE_NOTVIDEO)
    {
      oss.str("");
      oss << fileNameArray[i1-2] << "_map" << ".png";
      imwrite(oss.str(),map);

      oss.str("");
      oss << fileNameArray[i1-2] << "_mapBlue" << ".png";
      imwrite(oss.str(),mapBlue);

      oss.str("");
      oss << fileNameArray[i1-2] << "_mapLocalMax" << ".png";
      imwrite(oss.str(),mapLocalMax);
      
      oss.str("");
      oss << fileNameArray[i1-2] << "_colormap" << ".png";
      imwrite(oss.str(),colormap);
    }

  oss.str("");
  oss << fileNameArray[i1-2] << "_Hcomponents" << ".png";
  imwrite(oss.str(),h_components);

  oss.str("");
  oss << fileNameArray[i1-2] << "_Hthresh" << ".png";
  imwrite(oss.str(),h_thresh);

  oss.str("");
  oss << fileNameArray[i1-2] << "_Hmorph" << ".png";
  imwrite(oss.str(),h_morph);


  //local max images:
  oss.str("");
  oss << fileNameArray[i1-2] << "_maxSummed" << ".png";
  imwrite(oss.str(),histSummed);

  oss.str("");
  oss << fileNameArray[i1-2] << "_maxSummedGreen" << ".png";
  imwrite(oss.str(),histSummedGreen);

  oss.str("");
  oss << fileNameArray[i1-2] << "_maxComponents" << ".png";
  imwrite(oss.str(),componentsLocalMax);
  

  //blue images:
  oss.str("");
  oss << fileNameArray[i1-2] << "_blueThreshHigh" << ".png";
  imwrite(oss.str(),histBlue1);

  oss.str("");
  oss << fileNameArray[i1-2] << "_blueThreshLow" << ".png";
  imwrite(oss.str(),histBlue2);

  oss.str("");
  oss << fileNameArray[i1-2] << "_blueThreshSum" << ".png";
  imwrite(oss.str(),histThreshBlue);

  oss.str("");
  oss << fileNameArray[i1-2] << "_blueDilate" << ".png";
  imwrite(oss.str(),dilateBlue);

  oss.str("");
  oss << fileNameArray[i1-2] << "_blueErode" << ".png";
  imwrite(oss.str(),erodeBlue);

  oss.str("");
  oss << fileNameArray[i1-2] << "_blueComponents" << ".png";
  imwrite(oss.str(),componentsBlue);




  


  /*
  imwrite("map.png",map);
  imwrite("mapcolor.png",colormap);
  imwrite("outComponents.png",h_components);
  imwrite("outHthresh.png",h_thresh);
  imwrite("outHmorph.png",h_morph);
  */
  waitKey(0);
  outfile.close();
  return 0;



}
