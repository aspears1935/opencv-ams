/*
 * File Sonar Example using OpenCV
 * Demonstrate opening a file, accessing a head, and retriving a ping.
 * The ping is then processed into an image and displayed using OpenCV
 * Finally, a colormap is loaded and the image is colormapped.  The
 * color image is also displayed with OpenCV
 */

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>

#include <string.h>
#include <cv.h>
#include "opencv2/photo/photo.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include <highgui.h>
#include <cmath>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>

 using namespace std;
 using namespace cv;

 #define WRITE_FRAME_NUM true
// #define SYNCH_OFFSET_SEC -0.77411 //Offset between sonar and video data (video leading)
//DEBUG::::
//#define SYNCH_OFFSET_SEC -0 //Offset between sonar and video data (video leading)
//END DEBUG
 #define PI 3.14159265359
 #define DEPTH_OFFSET 60
 #define max_BINARY_value 255
 // #define VID_FOV_X 85

//double camArr30degWin[3][3] = {471.272, 0, 359.5,0, 468.339, 269.5, 0,0,1};
//double camArr30degWin[3][3] = {471.272, 0, 359.5,0, 468.339, 239.5, 0,0,1}; //Run11
//double distArr30degWin[5][1] = {-0.318162, 0.137179, 0.00494844, 0.00176499, -0.040086};

 // double camArr0degWin[3][3] = {450.247, 0, 359.5,0, 452.456, 269.5, 0,0,1};
 //double distArr0degWin[5][1] = {-0.297356, 0.11297, 0.00309039, 0.00158565, -0.0248698};

 RNG rng(12345);

 //Variable to save trackbar frame number
 int i = 0;
 bool tbar_update = false;

 void Trackbar(int, void*)
 {
   tbar_update = true;
 }


 int main( int argc, char *argv[] )
 {
 	bool pause = false; 
 	char keypushed = 0;
 	int currframe = 0;
	// 	int VID_PIX_X;

      if (argc < 2) {
 	  printf("usage: ./sonar_opencv <video-file>\n");
 	  printf("example: ./sonar_opencv ../../../data/swimmer.avi\n");
 	  exit(-1);
      }

      char videoFileName[256];
      strcpy(videoFileName, argv[1]);

      char video_wnd[] = "Video: 'b'=back, 'f'=forward, 'p'=pause, 'ESC'=exit";
      namedWindow(video_wnd,1);

      cout << "Opening input video..." << endl;
      cv::VideoCapture inputVideo(videoFileName);
      if(!inputVideo.isOpened())
	{
	  std::cout << "Could not open the input video: " << videoFileName << std::endl;
	  return -1;
	}
      
      //Get file creation date/time
      struct tm* clock;
      struct stat attrib;
      stat(videoFileName, &attrib);
      clock = gmtime(&(attrib.st_mtime));
      cout << "Last Modified: \n Date:" <<  clock->tm_mon+1 << "-" << clock->tm_mday << "-" << clock->tm_year+1900 << endl;
      cout << "Time: " << clock->tm_hour << ":" << clock->tm_min << ":" << clock->tm_sec << endl;
      
      char date[10];
      strftime(date, 10, "%d-%m-%y", clock);
      cout << date << endl;

      cv::Size videoSize = cv::Size((int) inputVideo.get(CV_CAP_PROP_FRAME_WIDTH),
				    (int) inputVideo.get(CV_CAP_PROP_FRAME_HEIGHT));

	std::cout << "Input frame resolution: Width=" << videoSize.width << "  Height=" << videoSize.height << " Num of Frames: " << inputVideo.get(CV_CAP_PROP_FRAME_COUNT) << " FPS=" << inputVideo.get(CV_CAP_PROP_FPS) << std::endl;
	cv::Mat videoImg;

	double VIDEO_FPS = inputVideo.get(CV_CAP_PROP_FPS);
	double VIDEO_FRAME_COUNT = inputVideo.get(CV_CAP_PROP_FRAME_COUNT);

	//Create Trackbars:             
	int max_frame_num = VIDEO_FRAME_COUNT-1;
        char position_tbar[] = "Position (0 based)";
	cv::createTrackbar(position_tbar, video_wnd, &i, max_frame_num, Trackbar);

	//---------Create Video Writer
	/*int ex = static_cast<int>(inputVideo.get(CV_CAP_PROP_FOURCC)); 
	char EXT[] = {(char)(ex & 0XFF) , (char)((ex & 0XFF00) >> 8),(char)((ex & 0XFF0000) >> 16),(char)((ex & 0XFF000000) >> 24), 0};
	cout << EXT << endl;
	cout << EXT[3] << EXT[2] << EXT[1] << EXT[0] << endl;
	cout << inputVideo.get(CV_CAP_PROP_FOURCC) << endl;
	int writerFPS = 20;
	cv::VideoWriter outputVideo;
	cv::VideoWriter outputSonar;
	outputVideo.open ("outputVideo.avi", CV_FOURCC('D','I','V','X'), writerFPS, videoSize, true );
	//outputSonar.open ("outputSonar.avi", CV_FOURCC('D','I','V','X'), writerFPS, Size(width,height), true );
	*/
	//---------------------------------------------------------------------
	//----------------Main Loop--------------------------------------------

	for (i = 0; i < VIDEO_FRAME_COUNT; i++) 
	  {
	    
	    inputVideo.set(CAP_PROP_POS_FRAMES,i);
	    inputVideo >> videoImg;

	    if(videoImg.empty())
	     	std::cout << "Frame " << i /*inputVideo.get(CAP_PROP_POS_FRAMES)*/ << "/" << VIDEO_FRAME_COUNT << " (0-based) - EMPTY FRAME" << std::endl;
	    else	    
	      {
		//NOTE: i is 0-based, get(POS_FRAMES) is 1-based
		std::cout << "Frame " << i /*inputVideo.get(CAP_PROP_POS_FRAMES)*/ << "/" << VIDEO_FRAME_COUNT << " (0-based) - " << inputVideo.get(CAP_PROP_POS_MSEC)/1000 - (1/VIDEO_FPS) << " sec - " << (int)(inputVideo.get(CAP_PROP_POS_MSEC)/3600000) << ":" << (int)(inputVideo.get(CAP_PROP_POS_MSEC)/60000)%60 << ":" << (int)(inputVideo.get(CAP_PROP_POS_MSEC)/1000)%60  << std::endl;
		//cout << i << "--" << inputVideo.get(CAP_PROP_POS_FRAMES) << endl;
	    
		if(WRITE_FRAME_NUM)
		  {
		    //Put Frame Number on image
		    char text[255];
		    sprintf(text, "%d", i);
		    cv::putText(videoImg, text, cvPoint(0,25), cv::FONT_HERSHEY_SIMPLEX,1,cv::Scalar::all(255));
		  }
		
		//Display Images:
		cv::imshow(video_wnd, videoImg);
	      }

	    //------------------------------------
	    //Check for key press:
	    if(pause)
	      {
		while(1){
		  keypushed = cvWaitKey(10);
		  if(keypushed==27)
		    break;
		  else if(keypushed=='p')
		    {
		      pause=false;
		      break;
		    }
		  else if(keypushed=='f')
		    {
		      //Go Forward 1 frame - do nothing because will be auto incremented
		      break;
		    }
		  else if(keypushed=='b')
		    {
		      i=i-2;	//Go back 2 frames (really going back one frame)
		      break;
		    }
		  /*		  else if(keypushed=='s')
		    {
		      imwrite("output.png",xxx);
		      }*/
		  else if(tbar_update)
		    {
		      tbar_update = false;
		      i=i-1;
		    break;
		    }
		}
		if(keypushed==27)
		  break;
	      }
	    else
	      {
		keypushed = cvWaitKey(10);
		if(keypushed==27)
		  break;
		else if(keypushed=='p')
		  {
		    pause=true;
		    i = i-1; //decrement because will be auto incremented back	
		  }	
	      }
	    //------------------------------------
     }
	
	return 0;
 }
