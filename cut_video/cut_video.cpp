
#include "opencv2/opencv.hpp"
#include "opencv2/photo/photo.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cmath>

/*#include "../shiftEstVideoOnly/my_ptsetreg.hpp"
#include "../shiftEstVideoOnly/my_five-point.hpp"
#include "../shiftEstVideoOnly/RANSAC.hpp"
#include "../shiftEstVideoOnly/precomp.hpp"

#include <cv.h>
#include <algorithm>
#include <iterator>
#include <limits>
#include "opencv2/core/utility.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/nonfree/nonfree.hpp"
*/

using namespace cv;
using namespace std;

/**
 * @function main
 */

int main( int argc, char** argv )
{
  //  RNG rng;

  if(argc < 4)
    {
      cout << "Not enough arguments" << endl;
      cout << "Usage: ./CutVideo <input video> <Start frame (0-based)> <End Frame>" << endl;
      cout << "Example: ./CutVideo input.mp4 0 100" << endl;
      return 0;
    }
  
  char videoFileName[256];
  strcpy(videoFileName, argv[1]);
  int startFrame = atoi(argv[2]);
  int endFrame = atoi(argv[3]);

  //  namedWindow("Sonar Rot Rect", WINDOW_AUTOSIZE);

  VideoCapture inputVideo(videoFileName);
  if(!inputVideo.isOpened())
    {
      cout << "Could not open the input video: " << videoFileName << endl;
      return -1;
    }

  Size videoSize = Size((int) inputVideo.get(CAP_PROP_FRAME_WIDTH), 
			(int) inputVideo.get(CAP_PROP_FRAME_HEIGHT));
  int inputNumFrames = inputVideo.get(CAP_PROP_FRAME_COUNT);
  int inputFPS =  inputVideo.get(CAP_PROP_FPS);
  int inputFourcc = inputVideo.get(CAP_PROP_FOURCC);
  char inputFourccChar[4];

  //Get char representation of FOURCC
  inputFourccChar[0] = inputFourcc%256;
  inputFourccChar[1] = (inputFourcc >> 8)%256;
  inputFourccChar[2] = (inputFourcc >> 16)%256;
  inputFourccChar[3] = (inputFourcc >> 24)%256;
  
  //cout << "avc1 - 828601953 - 0 x 31 63 76 61" << endl;

  cout << "Input frame resolution: Width=" << videoSize.width << "  Height=" << videoSize.height << " Num of Frames: " << inputNumFrames << " FPS=" << inputFPS << endl;
  cout << "Input FOURCC: " << inputFourccChar << " - " <<  inputFourcc << endl;

  VideoWriter outputVideo;

  const string outFileName = "output.avi";

  cout << "Output File Name=" << outFileName << endl;
  cout << "cv::VideoWriter::fourcc('X','V','I','D')=" << cv::VideoWriter::fourcc('X','V','I','D') << endl;

  //  outputVideo.open(outFileName, cv::VideoWriter::fourcc('X','V','I','D'), fps, Size(sonWidth,sonHeight),true); //CV_FOURCC('M','J','P','G')
  //Note: Can't use videowriter for some video types. For now, just use XVID
  outputVideo.open(outFileName, VideoWriter::fourcc('X','V','I','D')/*inputFourcc*/, inputFPS, videoSize, true); //CV_FOURCC('M','J','P','G')
  
  if (!outputVideo.isOpened())
    {
      cout  << "Could not open the output video for write: " << endl;
      return -1;
    }
  else
    cout  << "Opened the output video" << endl;

  int i = 0;
  for(i=startFrame; i<endFrame+1 ; i++)
    {
      Mat videoImg;
      inputVideo.set(CAP_PROP_POS_FRAMES, i);
      inputVideo >> videoImg;
      cout << i << "--" << inputVideo.get(CAP_PROP_POS_FRAMES) << endl;
      if(videoImg.empty())
	{
	  cout<<"^^Empty frame^^" << endl;
	  videoImg = Mat::zeros(videoSize.height,videoSize.width,CV_8UC3);
	}
            //imshow("INPUT", videoImg);
            //waitKey(0);
      outputVideo << videoImg;
    }
  
  cout << "Finished Writing" << endl;
  
  return 0;
}
