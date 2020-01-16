
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string.h>
#include <cmath>

#include <cv.h>
//#include <highgui.h>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/photo/photo.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

//-------------Main Function------------------//
 int main( int argc, char *argv[] )
 {

	cout << "Using OpenCV v" << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << "." << CV_SUBMINOR_VERSION << endl;

      if (argc < 3) {
 	  printf("usage: ./sonar_opencv <video-file> <frameNum>\n");
 	  printf("example: ./sonar_opencv ../../../Videos/test.avi 100\n");
 	  exit(-1);
      }

      char videoFileName[256];
      strcpy(videoFileName, argv[1]);
      int frameNum = (int)atoi(argv[2]);

	//-------------Open Input Video File---------------------
	cv::VideoCapture inputVideo(videoFileName);
	if(!inputVideo.isOpened())
	  {
	    std::cout << "Could not open the input video: " << videoFileName << std::endl;
	    return -1;
	  }

	namedWindow("VIDEO", WINDOW_AUTOSIZE);

	//-------------Get Video Properties-------------------
	cv::Size videoSize = cv::Size((int) inputVideo.get(CAP_PROP_FRAME_WIDTH),
				      (int) inputVideo.get(CAP_PROP_FRAME_HEIGHT));
	double VIDEO_FPS = inputVideo.get(CAP_PROP_FPS);
	double VIDEO_FRAME_COUNT = inputVideo.get(CAP_PROP_FRAME_COUNT);
	int numVidFrames = (int)VIDEO_FRAME_COUNT;
	std::cout << "Input frame resolution: Width=" << videoSize.width << "  Height=" << videoSize.height << " Num of Frames=" << numVidFrames << " FPS=" << inputVideo.get(CAP_PROP_FPS) << std::endl;

	//------------------Create Mat Structures---------------------
	Mat videoImg;

       	inputVideo.set(CAP_PROP_POS_FRAMES, frameNum);
	inputVideo >> videoImg;
	
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

	/*	istringstream iss(argv[1]);
	string filePrefix;
	getline(iss,filePrefix, '.');
	cout << filePrefix << endl;
	*/

	stringstream oss;

	oss << fileNameArray[i1-2] << "_frame" << frameNum << ".png";

	imwrite(oss.str(), videoImg);
	
	imshow("VIDEO", videoImg);

	waitKey(0);
    
	return 0;
}
