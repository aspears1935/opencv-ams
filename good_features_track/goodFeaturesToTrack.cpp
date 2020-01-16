/**
 * @function goodFeaturesToTrack_Demo.cpp
 * @brief Demo code for detecting corners using Shi-Tomasi method
 * @author OpenCV team
 */


//TODO:
//Make clusterCount a function of points detected?
//Plot a map like in the hsv_split code

//#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <fstream>

#define STOP_BETWEEN_FRAMES false
#define VERBOSE false
//Setup number one:  KMEANS=4, RADIUS=50, MIN_POINTS=70
//Setup number one:  KMEANS=10, RADIUS=30, MIN_POINTS=50
#define KMEANS 10 //Number of groups to divide into for kmeans
#define RADIUS 30 //Defines size of box to search for object groupings of points
#define MIN_POINTS_FOR_FEATURE 50 //Number of features in box to be interesting
#define MAP_SCALAR 10
#define IMAGE_NOTVIDEO false
#define WRITE_VIDEO true

#define STATIC_NUM_CLUSTERS false
#define CORNERS_PER_CLUSTER 10
#define CORNERS_PER_CLUSTER_SURF 20 //20
#define PCT_PER_CLUSTER 0.8
#define PCT_PER_CLUSTER_SURF 0.8

using namespace cv;
using namespace std;

/// Global variables
Mat src, src_gray;

int maxCorners = 1000;
int maxTrackbar = 1000;

RNG rng(12345);
const char* source_window = "Image";
VideoCapture cap;


/// Function header
void goodFeaturesToTrack_Demo( int, void* );

/**
 * @function main
 */
int main( int argc, char** argv )
{
  /// Load source image and convert it to gray
  //  src = imread( argv[1], 1 );
  VideoCapture cap;
  double frames;
  int height, width;

  if(argc < 2)
    {
      cout << "Not Enough Args. Usage: ./opencv <video_file>" << endl;
      return -1;
    }
  else
    {
      if(IMAGE_NOTVIDEO)
	{
	  src= imread(argv[1], IMREAD_COLOR);
	  frames = 1;
	  height = src.rows;
	  width = src.cols;
	  cout << "Height x Width = " << height << "x" << width << endl;
	  cout << "Area = " << height*width << endl;
	}
      else //Input is a video
	{
	  cap = VideoCapture(argv[1]);
	  if(!cap.isOpened())
	    {
	      cout << "CANNOT OPEN INPUT VIDEO" << endl;
	      return -1;
	    }
	  frames = (int)cap.get(CAP_PROP_FRAME_COUNT);
	  height = cap.get(CAP_PROP_FRAME_HEIGHT);
	  width = cap.get(CAP_PROP_FRAME_WIDTH);
	}
    }

  string fileNameArray[10];
  char lastStr[256];
  char * pch;
  pch = strtok(argv[1],"./");
  int i3 = 0;
  while(pch != NULL)
    {
      string tempstr(pch);
      strcpy(lastStr,tempstr.c_str());
      fileNameArray[i3] = lastStr;
      i3++;
      pch = strtok(NULL, "./");
    }


  double fx = 1050; //Blender=1050                                                  
  double fy = 1050; //Blender=1050                                                  
  double altitude = 20; //Blender - estimate 5meters to ice?                        
  double posx = 100;
  double posy = 100;

  double FOVX = 2*(180/CV_PI)*atan((width/2)/fx);
  cout << "FOVX=" << FOVX << endl;
  double FOVY = 2*(180/CV_PI)*atan((height/2)/fy);
  cout << "FOVY=" << FOVY << endl;

  ofstream outfile("output.csv");
  outfile << "frame;numOFcorners;OFmeanx;OFmeany;OFsdevx;OFsdevy;OFcover;";
  outfile << "numSURFcorners;SURFmeanx;SURFmeany;SURFsdevx;SURFsdevy;SURFcover;Bothcover;ObjectDetected;ObjectDetectedSURF;" << endl;

  //  cap >> src;
  //cvtColor( src, src_gray, COLOR_BGR2GRAY );

  /// Create Window
  namedWindow( source_window, WINDOW_AUTOSIZE );
  namedWindow( "Output Video", WINDOW_AUTOSIZE );

  /// Create Trackbar to set the number of corners
  createTrackbar( "Max  corners:", source_window, &maxCorners, maxTrackbar, goodFeaturesToTrack_Demo );

  //imshow( source_window, src );

  int minHessian = 400;
  SurfFeatureDetector vidSURFdetector(minHessian);
  std::vector<KeyPoint> vidSURFkeypoints;

  Mat map = Mat::zeros(800,1200,CV_8UC3);
  Mat mapTexture = Mat::zeros(800,1200,CV_8UC3);
  Mat mapTextureBoth = Mat::zeros(800,1200,CV_8UC3);
  namedWindow( "MAP", WINDOW_AUTOSIZE );
  namedWindow( "MAP TEXTURE", WINDOW_AUTOSIZE );


  VideoWriter outputVideoTexture;
  VideoWriter outputVideoTextureAnomaly;
  int outputFPS = 10;
  if(WRITE_VIDEO)
    {
      outputVideoTexture.open("outputTexture.avi",cv::VideoWriter::fourcc('X','V','I','D'), outputFPS, Size(width*2,height*2),true);
      if(!outputVideoTexture.isOpened())
	{
	  cout << "Could not open the output video for write" << endl;
	  return -1;
	}

      outputVideoTextureAnomaly.open("outputTextureAnomaly.avi",cv::VideoWriter::fourcc('X','V','I','D'), outputFPS, Size(width*2,height*2),true);
      if(!outputVideoTextureAnomaly.isOpened())
	{
	  cout << "Could not open the output video for write" << endl;
	  return -1;
	}
    }

  Mat featureImgOF, featureImgSURF, featureImgBoth, kmeansImgOF, kmeansImgSURF, redBoxesOF, redBoxesSURF;

  int i=0;
  for(i=0;i<frames;i++)
    {
      cout << i+1 << "/" << frames << endl;

      //Flag for detected object:
      bool OBJECT_DETECTED = false;
      bool OBJECT_DETECTED_SURF = false;
      
      //goodFeaturesToTrack_Demo( 0, 0 );
      if( maxCorners < 1 ) { maxCorners = 1; }
      
      /// Parameters for Shi-Tomasi algorithm
      vector<Point2f> corners;
      double qualityLevel = 0.1; //0.1; //5-29 changed from 0.05 //0.01;
      double minDistance = 1;
      int blockSize = 3;
      bool useHarrisDetector = false;
      double k = 0.04;

      if(!IMAGE_NOTVIDEO)
	cap >> src;

      cvtColor( src, src_gray, COLOR_BGR2GRAY );
      
      /// Copy the source image
      Mat copy, surf, pointsBoth, pointCover, pointCoverSURF, pointCoverBoth;
      copy = src.clone();
      pointCover = Mat::zeros(src.rows,src.cols,CV_8U);
      pointCoverSURF = Mat::zeros(src.rows,src.cols,CV_8U);
      surf = src.clone();
      pointsBoth = src.clone();

      /// Apply corner detection
      goodFeaturesToTrack( src_gray,
			   corners,
			   maxCorners,
			   qualityLevel,
			   minDistance,
			   Mat(),
			   blockSize,
			   useHarrisDetector,
			   k );
      for(int i1 = 0; i1 < corners.size(); i1++)
	{
	  circle( copy, corners[i1], 4,  Scalar(255,0,0), -1);
	}
      featureImgOF = copy.clone(); 
      featureImgBoth = copy.clone(); 

      //SURF-----------------:
      vector<Point2f> vidSURFcorners;
      vidSURFdetector.detect(src_gray, vidSURFkeypoints);
      for(int i1 = 0; i1 < vidSURFkeypoints.size(); i1++)
	{
	  circle(surf,vidSURFkeypoints[i1].pt,4,Scalar(255,0,0), -1);
	  vidSURFcorners.push_back(vidSURFkeypoints[i1].pt);

	  circle(featureImgBoth,vidSURFkeypoints[i1].pt,4,Scalar(255,0,0), -1);
	}
      featureImgSURF = surf.clone(); 
      
      //Averages------------------:
      Scalar average;
      Scalar averageSURF;
      Mat meanOF, meanSURF, stddevOF, stddevSURF;

      if(corners.size() > 0)
	{
	  average = mean(corners);
	  meanStdDev(corners,meanOF,stddevOF);
	  if(VERBOSE)
	    {
	      cout << "Average of corners = " << average << endl;
	      cout << "OF MEAN:" << meanOF << endl << "OF STDDEV:" << stddevOF << endl;
	    }
	}
      else
	{
	  cout << "NO OF CORNERS!" << endl;
	  average = Scalar(0,0,0);
	  meanOF = Mat::zeros(1,3,CV_64F);
	  stddevOF = Mat::zeros(1,3,CV_64F);
	}

      if(vidSURFcorners.size() > 0)
	{
	  averageSURF = mean(vidSURFcorners);
	  meanStdDev(vidSURFcorners,meanSURF,stddevSURF);
	  if(VERBOSE)
	    {
	      cout << "Average of SURF corners = " << averageSURF << endl;
	      cout << "SURF MEAN:" << meanSURF << endl << "SURF STDDEV:" << stddevSURF << endl;
	    }
	}
      else
	{
	  cout << "NO SURF CORNERS!" << endl;
	  averageSURF = Scalar(0,0,0);
	  meanSURF = Mat::zeros(1,3,CV_64F);
	  stddevSURF = Mat::zeros(1,3,CV_64F);
	}

      /// Draw corners detected
      if(VERBOSE)
	{
	  cout << "Number of corners detected: " << corners.size() <<endl;
	  cout << "Number of SURF corners detected: " << vidSURFkeypoints.size() <<endl;
	}
      
      int clusterCount;
      int attempts=3;
      int r = 4;
      Mat labels, centers;
      int sumPointCoverOF=0;
      Scalar colorTab[] = 
	{
	  Scalar(0,0,255),
	  Scalar(0,255,0),
	  Scalar(255,100,100),
	  Scalar(255,0,255),
	  Scalar(0,255,255),
	  Scalar(255,0,0),
	  Scalar(100,100,255),
	  Scalar(100,255,100),
	  Scalar(100,200,200),
	  Scalar(200,200,100)
	};

      if(corners.size() > 0)
	{
	  int numCorners = corners.size();
	  if(VERBOSE)
	    cout << "Num of Corners: " << numCorners << endl;

	      int minPointsForFeature;
	      int boxRadius;

	      if(STATIC_NUM_CLUSTERS)
		{
		  clusterCount=KMEANS;
		  minPointsForFeature = MIN_POINTS_FOR_FEATURE;	      
		}
	      else
		{		  
		  if(corners.size() < 10)
		    {		 
		      clusterCount = corners.size();
		      minPointsForFeature = corners.size();
		    }
		  else
		    {
		      clusterCount = numCorners/CORNERS_PER_CLUSTER;
		      minPointsForFeature = clusterCount*PCT_PER_CLUSTER;
		    }		
		}

	      if(VERBOSE)
		cout << "Cluster Count = " << clusterCount << endl;

	      vector<Point2f> cornerGroups[clusterCount]; //create an array of vectors. one for each group

	  kmeans(corners, clusterCount, labels,
		 TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0),
		 attempts, KMEANS_PP_CENTERS, centers);
 
	  for( size_t i = 0; i < corners.size(); i++ )
	    { 
	      int clusterIdx = labels.at<int>(i);
	      circle( copy, corners[i], r, colorTab[clusterIdx%10], -1, 8, 0 ); 
	      circle( pointsBoth, corners[i], r, colorTab[clusterIdx%10], -1, 8, 0 ); 
	      circle( pointCover, corners[i], 20, Scalar(255,255,255), -1, 8, 0 ); 
	      //circle( copy, corners[i], r, Scalar(rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255)), -1, 8, 0 );
	      //Put the points into their groups:
	      cornerGroups[labels.at<int>(i)].push_back(corners[i]); 
	    }

	  kmeansImgOF = copy.clone();
	  redBoxesOF = copy.clone();
	  
	  Mat groupMeans[clusterCount];
	  Mat groupStddevs[clusterCount];
	  for(int i1=0; i1< clusterCount; i1++)
	    {
	      if(cornerGroups[i1].size() > 0)
		{
		  meanStdDev(cornerGroups[i1],groupMeans[i1],groupStddevs[i1]);
		}	      
	    }	    

	  if(VERBOSE)
	    {
	      for(int i1=0;i1 < clusterCount; i1++)
		{
		  //cout << "GROUP MEANS= " << groupMeans[i1] << endl;
		  cout << i1 << " - STDDEV = " << groupStddevs[i1] << endl;
		}
	    }
	  //  cout << "CENTERS=" << centers << endl;

	  //Find Interesting Groupings (many points close together)
	  for( int k=0; k<clusterCount; k++)
	    {
	      int groupingCount = 0;
	      for( size_t i1 = 0; i1 < corners.size(); i1++ )
		{ 
		  if((corners[i1].x < centers.at<Point2f>(k).x+RADIUS)&&(corners[i1].x > centers.at<Point2f>(k).x-RADIUS)&&(corners[i1].y < centers.at<Point2f>(k).y+RADIUS)&&(corners[i1].y > centers.at<Point2f>(k).y-RADIUS))
		    groupingCount++; //If within the box, add 1 to the count
		}

	      if(VERBOSE)
		cout << k << " - Grouping Count=" << groupingCount << endl;
	    
	      Scalar color;

	      if(groupingCount > minPointsForFeature)
		{
		  OBJECT_DETECTED = true;

		  color = Scalar(0,255,0);
		  rectangle(copy,Rect(centers.at<Point2f>(k).x-RADIUS, centers.at<Point2f>(k).y-RADIUS, RADIUS*2, RADIUS*2), color, 2);
		  rectangle(redBoxesOF,Rect(centers.at<Point2f>(k).x-RADIUS, centers.at<Point2f>(k).y-RADIUS, RADIUS*2, RADIUS*2), color, 2);

		  //DRAW MAP:
		  double xmap = altitude*tan((centers.at<Point2f>(k).x-(width/2))*(FOVX/width)*(CV_PI/180));
		  double ymap = altitude*tan((centers.at<Point2f>(k).y-(height/2))*(FOVY/height)*(CV_PI/180));
		  circle(map, Point(posx+(xmap*MAP_SCALAR),posy+(ymap*MAP_SCALAR)), RADIUS/MAP_SCALAR, Scalar(255,0,0),-1);
		}
	      else
		{
		  color = Scalar(0,0,255);
		  rectangle(redBoxesOF,Rect(centers.at<Point2f>(k).x-RADIUS, centers.at<Point2f>(k).y-RADIUS, RADIUS*2, RADIUS*2), color, 2);
		}
	      
	    }  

	  sumPointCoverOF = (int)sum(pointCover)[0]/255;
	  if(VERBOSE)
	    cout << "NUMBER OF PIXELS OPEN = " << sumPointCoverOF << endl;
	}

      //SURF:
      int clusterCountSURF=KMEANS;
      int sumPointCoverSURF=0;
      int sumPointCoverBoth=0;
      if(vidSURFcorners.size() < 10)
	clusterCountSURF=vidSURFcorners.size();
      Mat labelsSURF, centersSURF;
      if(vidSURFcorners.size() > 0)
	{
	  int minPointsForFeatureSURF;
	  int boxRadiusSURF;
	  
	  if(STATIC_NUM_CLUSTERS)
	    {
	      clusterCount=KMEANS;
	      minPointsForFeatureSURF = MIN_POINTS_FOR_FEATURE;	      
	    }
	  else
	    {		  
	      if(vidSURFcorners.size() < CORNERS_PER_CLUSTER_SURF)
		{		 
		  clusterCountSURF = 1;
		  minPointsForFeatureSURF = vidSURFcorners.size();
		}
	      else
		{
		  clusterCountSURF = vidSURFcorners.size()/CORNERS_PER_CLUSTER_SURF;
		  minPointsForFeatureSURF = clusterCountSURF*PCT_PER_CLUSTER_SURF;
		}		
	    }
	  
	  if(VERBOSE)
	    cout << "SURF Cluster Count, Min Points for Feature = " << clusterCountSURF << ", " << minPointsForFeatureSURF << endl;
	  
	  kmeans(vidSURFcorners, clusterCountSURF, labelsSURF,
		 TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0),
		 attempts, KMEANS_PP_CENTERS, centersSURF);
	  
	  for( size_t i = 0; i < vidSURFcorners.size(); i++ )
	    { 
	      int clusterIdx = labelsSURF.at<int>(i);
	      circle( surf, vidSURFcorners[i], r, colorTab[clusterIdx%10], -1, 8, 0 ); 
	      circle( pointsBoth, vidSURFcorners[i], r, colorTab[clusterIdx%10], -1, 8, 0 ); 
	      circle( pointCoverSURF, vidSURFcorners[i], 20, Scalar(255,255,255), -1, 8, 0 ); 
	      //circle( copy, corners[i], r, Scalar(rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255)), -1, 8, 0 ); 
	    }
	  sumPointCoverSURF = (int)sum(pointCoverSURF)[0]/255;	  
	  if(VERBOSE)
	    cout << "NUMBER OF PIXELS OPEN SURF = " << sumPointCoverSURF << endl;
	  
	  kmeansImgSURF = surf.clone();
	  redBoxesSURF = surf.clone();

	  for( int k=0; k<clusterCountSURF; k++)
	    {
	      int groupingCount = 0;
	      for( size_t i1 = 0; i1 < vidSURFcorners.size(); i1++ )
		{ 
		  if((vidSURFcorners[i1].x < centersSURF.at<Point2f>(k).x+RADIUS)&&(vidSURFcorners[i1].x > centersSURF.at<Point2f>(k).x-RADIUS)&&(vidSURFcorners[i1].y < centersSURF.at<Point2f>(k).y+RADIUS)&&(vidSURFcorners[i1].y > centersSURF.at<Point2f>(k).y-RADIUS))
		    groupingCount++; //If within the box, add 1 to the count
		}

	      if(VERBOSE)
		cout << k << " - SURF Grouping Count=" << groupingCount << endl;
	    
	      Scalar color;

	      if(groupingCount > minPointsForFeatureSURF)
		{
		  OBJECT_DETECTED_SURF = true;

		  color = Scalar(0,255,0);
		  rectangle(surf,Rect(centersSURF.at<Point2f>(k).x-RADIUS, centersSURF.at<Point2f>(k).y-RADIUS, RADIUS*2, RADIUS*2), color, 2);
		  rectangle(redBoxesSURF,Rect(centersSURF.at<Point2f>(k).x-RADIUS, centersSURF.at<Point2f>(k).y-RADIUS, RADIUS*2, RADIUS*2), color, 2);
		}
	      else
		{
		  color = Scalar(0,0,255);
		  rectangle(redBoxesSURF,Rect(centersSURF.at<Point2f>(k).x-RADIUS, centersSURF.at<Point2f>(k).y-RADIUS, RADIUS*2, RADIUS*2), color, 2);
		}
	    }
	}
    
      //AMS: Put in center point
      //circle( copy, Point2f(average[0],average[1]), 10, Scalar(255,255,255), -1, 8, 0 );

	  //Find combined point cover:
	  add(pointCover, pointCoverSURF, pointCoverBoth);
	  sumPointCoverBoth = (int)sum(pointCoverBoth)[0]/255;

	  cout << "Point Cover OF: " << sumPointCoverOF << endl;
	  cout << "Point Cover SURF: " << sumPointCoverSURF << endl;
	  cout << "Point Cover Both: " << sumPointCoverBoth << endl;

      outfile << i << ";" << corners.size() << ";" << meanOF.at<double>(0,0) << ";" << meanOF.at<double>(0,1) << ";" << stddevOF.at<double>(0,0) << ";" << stddevOF.at<double>(0,1) << ";" << sumPointCoverOF << ";";
      outfile << vidSURFcorners.size() << ";" << meanSURF.at<double>(0,0) << ";" <<  meanSURF.at<double>(0,1) << ";" << stddevSURF.at<double>(0,0) << ";" << stddevSURF.at<double>(0,1) << ";" << sumPointCoverSURF << ";" << sumPointCoverBoth << ";" << OBJECT_DETECTED << ";" << OBJECT_DETECTED_SURF << endl;

      //Plot vehicle location: (x=100m, y=40m, z=20m)
      Scalar green(0,255,0);
      Scalar red(0,0,255);
      if(OBJECT_DETECTED)
	circle(map, Point(posx, posy), 2, green);
      else
	circle(map, Point(posx, posy), 2, red);

      //Plot texture:
      double area = height*width;
      circle(mapTexture, Point(posx, posy), 2, Scalar(0, 255*(sumPointCoverSURF/area), 255-(255*(sumPointCoverSURF/area))),-1); // 255*(sumPointCoverSURF/area)), -1);
      circle(mapTextureBoth, Point(posx, posy), 2, Scalar(255*(sumPointCoverBoth/area), 255*(sumPointCoverBoth/area), 255*(sumPointCoverBoth/area)),-1); // 255*(sumPointCoverSURF/area)), -1);

      //Reset OBJECT_DETECTED flag:
      OBJECT_DETECTED = false;
      OBJECT_DETECTED_SURF = false;

      /// Show images
      namedWindow( source_window, WINDOW_AUTOSIZE );
      imshow( source_window, copy );
      namedWindow( "point cover", WINDOW_AUTOSIZE );
      imshow( "point cover", pointCover );
      namedWindow( "surf", WINDOW_AUTOSIZE );
      imshow( "surf", surf );
      namedWindow( "point cover SURF", WINDOW_AUTOSIZE );
      imshow( "point cover SURF", pointCoverSURF );
      
      imshow("MAP",map);
      imshow("MAP TEXTURE",mapTexture);

      //Write output video frame
      Mat outVideoImg = Mat::zeros(src.rows*2,src.cols*2,CV_8UC3);
      Mat tmp1, tmp2, tmp3, tmp4, tmp5;

      resize(src, tmp1, Size(width,height));
      Mat mapRoi1(outVideoImg, Rect(0, 0, src.cols, src.rows));
      tmp1.copyTo(mapRoi1);

      resize(pointsBoth, tmp2, Size(width,height));
      Mat mapRoi2(outVideoImg, Rect(src.cols, 0, src.cols, src.rows));
      tmp2.copyTo(mapRoi2);

      cvtColor(pointCoverBoth,tmp4,COLOR_GRAY2BGR);
      resize(tmp4, tmp3, Size(width,height));
      Mat mapRoi3(outVideoImg, Rect(0, src.rows, src.cols, src.rows));
      tmp3.copyTo(mapRoi3);

      //cvtColor(mapTextureBoth,tmp5,COLOR_GRAY2BGR);
      resize(mapTextureBoth, tmp4, Size(width,height));
      Mat mapRoi4(outVideoImg, Rect(src.cols, src.rows, src.cols, src.rows));
      tmp4.copyTo(mapRoi4);

      if(WRITE_VIDEO)
	outputVideoTexture << outVideoImg;

      //Write Anomaly Video:
      outVideoImg = Mat::zeros(src.rows*2,src.cols*2,CV_8UC3);

      resize(src, tmp1, Size(width,height));
      //      Mat mapRoi1(outVideoImg, Rect(0, 0, src.cols, src.rows));
      tmp1.copyTo(mapRoi1);

      resize(pointsBoth, tmp2, Size(width,height));
      //      Mat mapRoi2(outVideoImg, Rect(src.cols, 0, src.cols, src.rows));
      tmp2.copyTo(mapRoi2);

      resize(copy, tmp3, Size(width,height));
      //      Mat mapRoi3(outVideoImg, Rect(0, src.rows, src.cols, src.rows));
      tmp3.copyTo(mapRoi3);

      //cvtColor(mapTextureBoth,tmp5,COLOR_GRAY2BGR);
      resize(map, tmp4, Size(width,height));
      //      Mat mapRoi4(outVideoImg, Rect(src.cols, src.rows, src.cols, src.rows));
      tmp4.copyTo(mapRoi4);

      if(WRITE_VIDEO)
	outputVideoTextureAnomaly << outVideoImg;
      //      imshow( "Output Video", outVideoImg);
      //      waitKey(0);

      /*      istringstream iss(argv[1]);
      string filePrefix;
      getline(iss, filePrefix, '.');
      stringstream oss;

      oss.str("");
      oss << "output" << filePrefix;
      cout << filePrefix << endl;
      if(!IMAGE_NOTVIDEO)
	oss << "_frame" << i;
      oss << ".png";
      */

      if(VERBOSE)
	cout << "Writing Output Images" << endl;

      stringstream oss;

      oss.str("");
      oss << "output" << fileNameArray[i3-2];
      //      if(!IMAGE_NOTVIDEO)
      //	oss << "_frame" << i;
      oss << ".png";
      imwrite(oss.str(),copy);

      oss.str("");
      oss << "cover" << fileNameArray[i3-2];
      //      if(!IMAGE_NOTVIDEO)
      //	oss << "_frame" << i;
      oss << ".png";
      imwrite(oss.str(),pointCover);

      oss.str("");
      oss << "outputSURF" << fileNameArray[i3-2];
      //      if(!IMAGE_NOTVIDEO)
      //	oss << "_frame" << i;
      oss << ".png";
      imwrite(oss.str(),surf);

      oss.str("");
      oss << "outputBoth" << fileNameArray[i3-2];
      //      if(!IMAGE_NOTVIDEO)
      //	oss << "_frame" << i;
      oss << ".png";
      imwrite(oss.str(),pointsBoth);

      oss.str("");
      oss << "coverSURF" << fileNameArray[i3-2];
      //     if(!IMAGE_NOTVIDEO)
      //	oss << "_frame" << i;
      oss << ".png";
      imwrite(oss.str(),pointCoverSURF);

      oss.str("");
      oss << "coverBoth" << fileNameArray[i3-2];
      //if(!IMAGE_NOTVIDEO)
      //	oss << "_frame" << i;
      oss << ".png";
      imwrite(oss.str(),pointCoverBoth);

      //-----------Write out for presentation:
      oss.str("");
      oss << "featuresOF" << fileNameArray[i3-2];
      //if(!IMAGE_NOTVIDEO)
      //	oss << "_frame" << i;
      oss << ".png";
      imwrite(oss.str(),featureImgOF);

      oss.str("");
      oss << "featuresSURF" << fileNameArray[i3-2];
      //if(!IMAGE_NOTVIDEO)
      //	oss << "_frame" << i;
      oss << ".png";
      imwrite(oss.str(),featureImgSURF);

      oss.str("");
      oss << "featuresBoth" << fileNameArray[i3-2];
      //if(!IMAGE_NOTVIDEO)
      //	oss << "_frame" << i;
      oss << ".png";
      imwrite(oss.str(),featureImgBoth);

      oss.str("");
      oss << "kmeansOF" << fileNameArray[i3-2];
      //if(!IMAGE_NOTVIDEO)
      //	oss << "_frame" << i;
      oss << ".png";
      imwrite(oss.str(),kmeansImgOF);

      oss.str("");
      oss << "kmeansSURF" << fileNameArray[i3-2];
      //if(!IMAGE_NOTVIDEO)
      //	oss << "_frame" << i;
      oss << ".png";
      imwrite(oss.str(),kmeansImgSURF);

      oss.str("");
      oss << "redBoxesOF" << fileNameArray[i3-2];
      //if(!IMAGE_NOTVIDEO)
      //	oss << "_frame" << i;
      oss << ".png";
      imwrite(oss.str(),redBoxesOF);

      oss.str("");
      oss << "redBoxesSURF" << fileNameArray[i3-2];
      //if(!IMAGE_NOTVIDEO)
      //	oss << "_frame" << i;
      oss << ".png";
      imwrite(oss.str(),redBoxesSURF);

      //imwrite("output.png", copy);
      //imwrite("outputCover.png", pointCover);
      //imwrite("outputSURF.png",surf);
      //imwrite("outputCoverSURF.png",pointCoverSURF);

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

  waitKey(0);
  outfile.close();

  if(!IMAGE_NOTVIDEO)
    {
      imwrite("map.png",map);
      imwrite("TextureMap.png",mapTexture);
      imwrite("TextureMapBoth.png",mapTextureBoth);
    }

  return(0);
}

/**
 * @function goodFeaturesToTrack_Demo.cpp
 * @brief Apply Shi-Tomasi corner detector
 */
void goodFeaturesToTrack_Demo( int, void* )
{
  
}

