/**
 * @function calcHist_Demo.cpp
 * @brief Demo code to use the function calcHist
 * @author
 */

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <string>

using namespace std;
using namespace cv;

#define MAX_COUNT 0.01 //0.010 //TOP PCT - 1% good for real, 10% good for sim data
#define MIN_COUNT 0.0001 //0.0001 //MIN PCT - 10%=0.1

#define VERBOSE false
#define STOP_BETWEEN_IMGS false
#define SHRINK_IMGS true

//#define MIN_BOX_RADIUS 10
#define MIN_CONTOUR_AREA 150 //500
#define MIN_CONTOUR_AREA_GREEN 600 //500
#define MIN_CONTOUR_AREA_BLUE 0.00006
#define BLUE_LOW 81
#define BLUE_HIGH 151
#define GREEN_LOW 60
#define GREEN_HIGH 82

RNG rng(12345);

/**
 * @function main
 */
int main( int, char** argv )
{
  Mat src, dst;
  //  ofstream outfile("output.csv");

  /// Load image
  src = imread( argv[1], 1 );

  if( !src.data )
    { return -1; }

  cout << argv[1] << ",";

  Mat hsv_frame;
  cvtColor(src, hsv_frame, COLOR_BGR2HSV);

  /// Separate the image in 3 places ( B, G and R )
  vector<Mat> hsv_planes;
  split( hsv_frame, hsv_planes );

  /// Establish the number of bins
  int histSize = 180; //was 256

  /// Set the ranges ( for B,G,R) )
  float range[] = { 0, 180 } ; //was 256
  const float* histRange = { range };

  bool uniform = true; bool accumulate = false;

  Mat h_hist, s_hist, v_hist;

  //DEBUG - delete!
  //  threshold(hsv_planes[0], hsv_planes[0], 90, 255, THRESH_TOZERO_INV);//Remove high vals //149
  //  threshold(hsv_planes[0], hsv_planes[0], 90, 179, THRESH_TOZERO_INV); 

  /// Compute the histograms:
  calcHist( &hsv_planes[0], 1, 0, Mat(), h_hist, 1, &histSize, &histRange, uniform, accumulate );
  calcHist( &hsv_planes[1], 1, 0, Mat(), s_hist, 1, &histSize, &histRange, uniform, accumulate );
  calcHist( &hsv_planes[2], 1, 0, Mat(), v_hist, 1, &histSize, &histRange, uniform, accumulate );

  //AMS:
  Mat h_histT;
  transpose(h_hist, h_histT);
  cout << h_histT << endl;
  //  cout << (int)hsv_planes[0].at<char>(0,0) << endl;
  //  outfile << argv[1] << ",";
  //  outfile << h_histT << endl;

  // Draw the histograms for B, G and R
  int hist_w = 720; //was 512 
  int hist_h = 400;
  int bin_w = cvRound( (double) hist_w/histSize );

  Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

  /// Normalize the result to [ 0, histImage.rows ]
  normalize(h_hist, h_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
  normalize(s_hist, s_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
  normalize(v_hist, v_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

  /// Draw for each channel
  for( int i = 1; i < histSize; i++ )
  {
      line( histImage, Point( bin_w*(i-1), hist_h - cvRound(h_hist.at<float>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(h_hist.at<float>(i)) ),
                       Scalar( 255, 0, 0), 2, 8, 0  );
      /*      line( histImage, Point( bin_w*(i-1), hist_h - cvRound(s_hist.at<float>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(s_hist.at<float>(i)) ),
                       Scalar( 0, 255, 0), 2, 8, 0  );
      line( histImage, Point( bin_w*(i-1), hist_h - cvRound(v_hist.at<float>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(v_hist.at<float>(i)) ),
                       Scalar( 0, 0, 255), 2, 8, 0  );*/
  }

  //Add in the color pallet at the bottom:
  Mat bigHistImg = Mat::zeros(600,720,CV_8UC3);
  Mat roi1(bigHistImg,Rect(0,0,720,400)); //x,y,width,height
  Mat roi2(bigHistImg,Rect(0,400,720,200));
  roi1 += histImage.clone();
  roi2 += imread("hue720x200.png",1);


  /// Display
  namedWindow("HSV histogram. H=blue, S=green, V=red", WINDOW_AUTOSIZE );
  imshow("HSV histogram. H=blue, S=green, V=red", bigHistImg );
  if(STOP_BETWEEN_IMGS)
    waitKey(0);
  else
    waitKey(10);

  //Check for local maximums:
  Mat histImgNew, morphImg;
  Mat histSummed = Mat::zeros(src.rows,src.cols,CV_8U);
  Mat histSummedGreen = Mat::zeros(src.rows,src.cols,CV_8U);

  namedWindow("Original", WINDOW_AUTOSIZE );
  namedWindow("Thresh", WINDOW_AUTOSIZE );
  namedWindow("Morph", WINDOW_AUTOSIZE );
  namedWindow("Components", WINDOW_AUTOSIZE );
  
  int MIN, MAX;
  MIN = MIN_COUNT*(src.cols*src.rows);
  MAX = MAX_COUNT*(src.cols*src.rows);

  if(VERBOSE)
    {
      cout << "width=" << src.cols << endl;
      cout << "height=" << src.rows << endl;
      cout << "area=" << src.rows*src.cols << endl;
      
      cout << "MIN=" << MIN << endl;
      cout << "MAX=" << MAX << endl;

      cout << "BLUE MIN CONTOUR AREA = " << (int)(MIN_CONTOUR_AREA_BLUE*src.rows*src.cols) << endl;
    }


  //------------------------Blue Thresholding-----------------------:
  Mat histThreshBlue, histBlue1, histBlue2;//151,81
  threshold(hsv_planes[0], histBlue1, BLUE_HIGH, 255, THRESH_BINARY);//High vals
  threshold(hsv_planes[0], histBlue2, BLUE_LOW, 255, THRESH_TOZERO_INV); //Rm 0 Vals, if src>=thresh, dst=0
  threshold(histBlue2, histBlue2, 1, 255, THRESH_BINARY); //Rm Low Vals
  histThreshBlue = histBlue1+histBlue2;
  Mat noblue_hist, noblue_histT;;
  calcHist( &histThreshBlue, 1, 0, Mat(), noblue_hist, 1, &histSize, &histRange, uniform, accumulate );
  transpose(noblue_hist, noblue_histT);
  //  cout << noblue_histT << endl;
  //  erode(histThreshBlue, histThreshBlue, getStructuringElement(MORPH_RECT,Size(3,3),Point(1,1)), Point(-1,-1),1);
  //dilate(histThreshBlue, histThreshBlue, getStructuringElement(MORPH_RECT,Size(3,3),Point(1,1)), Point(-1,-1),1);

  //------Blue Components------//
  //First open image:
  vector<vector<Point> > contoursBlue;
  vector<Vec4i> hierarchyBlue;
  Mat tmp1 = histThreshBlue.clone();
  findContours(tmp1,contoursBlue,hierarchyBlue,RETR_CCOMP,CHAIN_APPROX_SIMPLE);

  vector<vector<Point> > contours_polyBlue(contoursBlue.size());
  vector<Rect> boundRectBlue(contoursBlue.size());
  vector<Point2f> centerBlue(contoursBlue.size());
  vector<float> radiusBlue(contoursBlue.size());
  vector<double> areaBlue(contoursBlue.size());

  Mat componentsBlue = Mat::zeros(src.rows,src.cols,CV_8UC3);
  for(int count = 0; count < contoursBlue.size(); count++)   
    {
      approxPolyDP(Mat(contoursBlue[count]),contours_polyBlue[count], 3, true);
      areaBlue[count] = contourArea(contoursBlue[count]);

      if(areaBlue[count] < (int)(MIN_CONTOUR_AREA_BLUE*src.rows*src.cols))
	continue;
      if(VERBOSE)
	cout << "Blue component size: " << areaBlue[count] << endl;
      Scalar color = Scalar(rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255));
      drawContours(componentsBlue, contours_polyBlue, count, color, FILLED/*1*/, 8, vector<Vec4i>(),0,Point());
      //rectangle(components, boundRect[count].tl(), boundRect[count].br(), color, 2, 8, 0);
    }
  if(VERBOSE)
    cout << "Showing Blue Components" << endl;
  imshow("Components", componentsBlue);

  if(STOP_BETWEEN_IMGS)
    waitKey(0);
  else
    waitKey(10);
  //--------------------------------------------------------------------------


  //Add all local maximums
  for(int i=1; i<histSize-1; i++) //Have to go from 1:size-1 to leave 1 buffer
    {
      if((h_histT.at<float>(i-1)<h_histT.at<float>(i))&&(h_histT.at<float>(i)>h_histT.at<float>(i+1))) //local max
	{
	  if((h_histT.at<float>(i) < MIN)||(h_histT.at<float>(i) > MAX)) //Check if between thresholds
	    {
	      if(VERBOSE)
		{
		  cout << "I = " << i << " -- Skipping, out of threshold..." << endl;
		  cout << h_histT.at<float>(i-1) << "--" << h_histT.at<float>(i) << "--" << h_histT.at<float>(i+1) << endl; 
		}
	      continue; //If outside thresholds, don't calculate
	    }

	  if((i>BLUE_LOW)&&(i<BLUE_HIGH)) //Check if between thresholds was 95
	    {
	      //cout << "I = " << i << " -- Skipping, BLUE!!..." << endl;
	      //cout << h_histT.at<float>(i-1) << "--" << h_histT.at<float>(i) << "--" << h_histT.at<float>(i+1) << endl; 
	      continue; //If outside thresholds, don't calculate
	    }

	  if((i==10)||(i==30)||(i==60)||(i==165)||(i==170))
	    {
	      continue; //If 165, don't show - this is a weird harmonic of blue?
	    }

	  if(VERBOSE)
	    {
	      cout << "I=" << i << endl;
	      cout << h_histT.at<float>(i-1) << "--" << h_histT.at<float>(i) << "--" << h_histT.at<float>(i+1) << endl; 
	    }

	  threshold(hsv_planes[0], histImgNew, i+1, 255, THRESH_TOZERO_INV); //was i-1

	  if(i>1)
	    threshold(histImgNew, histImgNew, i-2, 255, THRESH_BINARY); //was i-1
	  else
	    threshold(histImgNew, histImgNew, 1, 255, THRESH_BINARY); //was i-1


	  //cout << "Sum: " << sum(histImgNew)[0] << endl;
	  if((i>GREEN_LOW)&&(i<GREEN_HIGH)) //Greens, close to blue
	    {
	      erode(histImgNew, histImgNew, getStructuringElement(MORPH_RECT,Size(3,3),Point(1,1)), Point(-1,-1),3);
	      dilate(histImgNew, histImgNew, getStructuringElement(MORPH_RECT,Size(3,3),Point(1,1)), Point(-1,-1),3);
	      histSummedGreen += histImgNew;	      
	    }
	  else
	    histSummed += histImgNew;

	  if(SHRINK_IMGS)
	    {
	      Mat histImgNewSmall, srcSmall,histSummedSmall;
	      resize(histImgNew, histImgNewSmall, Size(src.cols/2,src.rows/2));
	      resize(src, srcSmall, Size(src.cols/2,src.rows/2));
	      resize(histSummed, histSummedSmall, Size(src.cols/2,src.rows/2));
	      imshow("Thresh", histImgNewSmall );
	      imshow("Original", srcSmall);
	      imshow("Summed", histSummedSmall);
	    }
	  else
	    {
	      imshow("Thresh", histImgNew );
	      imshow("Original", src);
	      imshow("Summed", histSummed);
	    }

	  if(STOP_BETWEEN_IMGS)
	    waitKey(0);
	  else
	    waitKey(10);

	  //DEBUG _DELETE!
	  //	  dilate(histImgNew, histImgNew, getStructuringElement(MORPH_RECT,Size(3,3),Point(1,1)), Point(-1,-1),1);
	  //erode(histImgNew, histImgNew, getStructuringElement(MORPH_RECT,Size(3,3),Point(1,1)), Point(-1,-1),1);


	}
    }

  //------------------------Components:--------------------//
  //Mat components;
  //First open image:
  //dilate(histSummed, histSummed, getStructuringElement(MORPH_RECT,Size(3,3),Point(1,1)), Point(-1,-1),4);  
  //erode(histSummed, histSummed, getStructuringElement(MORPH_RECT,Size(3,3),Point(1,1)), Point(-1,-1),4);

  // erode(histSummed, histSummed, getStructuringElement(MORPH_RECT,Size(3,3),Point(1,1)), Point(-1,-1),2);
  // dilate(histSummed, histSummed, getStructuringElement(MORPH_RECT,Size(3,3),Point(1,1)), Point(-1,-1),2);
  


  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;
  //NOTE: Problem with weird gradient looking images stem from findContours(). It changes the input Mat. Maybe make a copy before sending it to this function.
  Mat tmp = histSummed.clone(); //Was histSummed
  findContours(tmp,contours,hierarchy,RETR_CCOMP,CHAIN_APPROX_SIMPLE);
  vector<vector<Point> > contours_poly(contours.size());
  vector<Rect> boundRect(contours.size());
  vector<Point2f> center(contours.size());
  vector<float> radius(contours.size());
  vector<double> area(contours.size());
  //  cvtColor(histSummed, components, COLOR_GRAY2BGR);
  Mat components = Mat::zeros(src.rows,src.cols,CV_8UC3);
  for(int count = 0; count < contours.size(); count++)
    {
      approxPolyDP(Mat(contours[count]),contours_poly[count], 3, true);
      boundRect[count] = boundingRect(Mat(contours_poly[count]));
      minEnclosingCircle((Mat)contours_poly[count], center[count], radius[count]);
      area[count] = contourArea(contours[count]);

      //      if(radius[count] < MIN_BOX_RADIUS)
      if(area[count] < MIN_CONTOUR_AREA)
      	continue;
      if(VERBOSE)
	cout << "Component Area: " << area[count] << endl;
      Scalar color = Scalar(rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255));
      drawContours(components, contours_poly, count, color, FILLED/*1*/, 8, vector<Vec4i>(),0,Point());
      //rectangle(components, boundRect[count].tl(), boundRect[count].br(), color, 2, 8, 0);
    }
  imshow("Components", components);
  if(STOP_BETWEEN_IMGS)
    waitKey(0);
  else
    waitKey(10);
  
  //------------------------Green Components:--------------------//
  //First open image:
  vector<vector<Point> > contoursGreen;
  vector<Vec4i> hierarchyGreen;
  tmp = histSummedGreen.clone();
  findContours(tmp,contoursGreen,hierarchyGreen,RETR_CCOMP,CHAIN_APPROX_SIMPLE);

  vector<vector<Point> > contours_polyGreen(contoursGreen.size());
  vector<Rect> boundRectGreen(contoursGreen.size());
  vector<Point2f> centerGreen(contoursGreen.size());
  vector<float> radiusGreen(contoursGreen.size());
  vector<double> areaGreen(contoursGreen.size());
  //  cvtColor(histSummed, components, COLOR_GRAY2BGR);
  Mat componentsGreen = Mat::zeros(src.rows,src.cols,CV_8UC3);
  for(int count = 0; count < contoursGreen.size(); count++)   
    {
      approxPolyDP(Mat(contoursGreen[count]),contours_polyGreen[count], 3, true);
      areaGreen[count] = contourArea(contoursGreen[count]);

      if(areaGreen[count] < MIN_CONTOUR_AREA_GREEN)
	continue;
      if(VERBOSE)
	cout << "Green Component Area: " << areaGreen[count] << endl;
      Scalar color = Scalar(rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255));
      drawContours(componentsGreen, contours_polyGreen, count, color, FILLED/*1*/, 8, vector<Vec4i>(),0,Point());
      //rectangle(components, boundRect[count].tl(), boundRect[count].br(), color, 2, 8, 0);
    }
  imshow("Components", componentsGreen);
  if(STOP_BETWEEN_IMGS)
    waitKey(0);
  else
    waitKey(10);
  //-----------------------------------------------------------//
  //Add in greens:
  components += componentsGreen;
  histSummed += histSummedGreen;
  
      //-----------------Mophological------------------:
      Mat morphImgSmall;      
      dilate(histSummed, morphImg, getStructuringElement(MORPH_RECT,Size(3,3),Point(1,1)), Point(-1,-1),6);
      
      if(SHRINK_IMGS)
	{
	  resize(morphImg, morphImgSmall, Size(src.cols/2,src.rows/2));  
	  imshow("Morph", morphImgSmall);
	}
      else
	imshow("Morph", morphImg);

      if(STOP_BETWEEN_IMGS)
	waitKey(0);
      
      
      erode(morphImg, morphImg, getStructuringElement(MORPH_RECT,Size(3,3),Point(1,1)), Point(-1,-1),7);
      
      if(SHRINK_IMGS)
	{
	  resize(morphImg, morphImgSmall, Size(src.cols/2,src.rows/2));  
	  imshow("Morph", morphImgSmall);
	}
      else
	imshow("Morph", morphImg);
      
      if(STOP_BETWEEN_IMGS)
	waitKey(0);
      else
	waitKey(10);
      
      erode(morphImg, morphImg, getStructuringElement(MORPH_RECT,Size(3,3),Point(1,1)), Point(-1,-1),1);
      dilate(morphImg, morphImg, getStructuringElement(MORPH_RECT,Size(3,3),Point(1,1)), Point(-1,-1),2);

  if(SHRINK_IMGS)
    {
      resize(morphImg, morphImgSmall, Size(src.cols/2,src.rows/2));  
      imshow("Morph", morphImgSmall);
    }
  else
    imshow("Morph", morphImg);

  if(STOP_BETWEEN_IMGS)
    waitKey(0);
  else
    waitKey(10);

  /*  char inFileNameChar[256];
  strcpy(inFileNameChar, argv[1]);

  istream inFileNameStream(argv[1]);

  stringstream filePrefix;
  getline(inFileNameStream,filePrefix,'.');
  cout << filePrefix << endl;
  */

  istringstream iss(argv[1]);
  string filePrefix;
  getline(iss, filePrefix, '.');
  //cout << filePrefix << endl;

  stringstream oss;

  oss.str(""); //clear stream
  oss << "outHistBlueHigh" << filePrefix << ".png";
  imwrite(oss.str(),histBlue1);

  oss.str(""); //clear stream
  oss << "outHistBlueLow" << filePrefix << ".png";
  imwrite(oss.str(),histBlue2);

  oss.str(""); //clear stream
  oss << "outThreshBlue" << filePrefix << ".png";
  imwrite(oss.str(),histThreshBlue);

  oss.str(""); //clear stream
  oss << "outComponentsBlue" << filePrefix << ".png";
  imwrite(oss.str(),componentsBlue);

  //-----------Now the max threshold-----------:
  oss.str(""); //clear stream
  oss << "outHist" << filePrefix << ".png";
  imwrite(oss.str(),bigHistImg);

  oss.str(""); //clear stream
  oss << "outSummed" << filePrefix << ".png";
  imwrite(oss.str(),histSummed);

  //  oss.str(""); //clear stream
  //oss << "outSummedGreen" << filePrefix << ".png";
  //imwrite(oss.str(),histSummedGreen);

  oss.str(""); //clear stream
  oss << "outMorph" << filePrefix << ".png";
  imwrite(oss.str(),morphImg);

  oss.str(""); //clear stream
  oss << "outComponents" << filePrefix << ".png";;
  imwrite(oss.str(),components);


  //outfile.close();
  return 0;

}
