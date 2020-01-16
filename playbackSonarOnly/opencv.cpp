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

#include <bvt_sdk.h>
#include <string.h>
#include <cv.h>
#include "opencv2/photo/photo.hpp"
#include <highgui.h>
#include <cmath>

 using namespace std;
 using namespace cv;

 #define WRITE_FRAME_NUM true

 #define PI 3.14159265359
 #define max_BINARY_value 255
 #define VERBOSE false

 RNG rng(12345);

 //Variable to save trackbar frame number
 int i = 0;
 bool tbar_update = false;
 bool Windows_Data = false; 

 void Trackbar(int, void*)
 {
   tbar_update = true;
 }


 int main( int argc, char *argv[] )
 {
 	bool pause = false; 
 	char keypushed = 0;
 	int currframe = 0;
 	double SONAR_TIME0 = 0;
 	double ping_time_sec = 0;  //Time of Ping from TIME0 in Seconds
	double min_rangeRes = 1000; //Meters/pixel
	double max_rangeRes = 0;
	double min_bearingRes = 1000; //Deg/column
	double max_bearingRes = 0;
	double min_height = 1000;
	double max_height = 0;
	double min_width = 1000;
	double max_width = 0;

      if (argc < 2) {
 	  printf("usage: ./sonar_opencv <sonar-file>\n");
 	  printf("example: ./sonar_opencv ../../../data/swimmer.son\n");
 	  exit(-1);
      }

      char DataFile[256];
      strcpy(DataFile, argv[1]);
      cout << DataFile << endl;

      ofstream outFile("outputRangeRes.csv");
      outFile << "frame;rangeResMag;bearingResMag;rangeResClr;height;width;" << endl;

      int ret;
      // Create a new BVTSonar Object
      BVTSonar son = BVTSonar_Create();
      if( son == NULL )
      {
 	  printf("BVTSonar_Create: failed\n");
 	  return 1;
      }
      // Open the sonar
      ret = BVTSonar_Open(son, "FILE", DataFile);
      if( ret != 0 )
      {
 	  printf("BVTSonar_Open: ret=%d\n", ret);
 	  return 1;
      }

      // Make sure we have the right number of heads
      int heads = -1;
      heads = BVTSonar_GetHeadCount(son);
      if(VERBOSE)
	printf("BVTSonar_GetHeadCount: %d\n", heads);

      // Get the first head
      BVTHead head = NULL;
      ret = BVTSonar_GetHead(son, 0, &head);
      if( ret != 0 )
      {
 	  printf("BVTSonar_GetHead: ret=%d\n", ret);
 	  return 1;
      }
	
      // Check the ping count
      int pings = -1;
      pings = BVTHead_GetPingCount(head);
      if(VERBOSE)
	printf("BVTHead_GetPingCount: %d\n", pings);

      // Set the range window to be 10m to 40m
      //BVTHead_SetRange(head, 0, 50);

      int height, width;
      int height_mag, width_mag;

      // Build a color mapper
      BVTColorMapper mapper;
      mapper = BVTColorMapper_Create();
      if( mapper == NULL )
      {
 	  printf("BVTColorMapper_Create: failed\n");
 	  return 1;
      }

      // Load the bone colormap
      ret = BVTColorMapper_Load(mapper, "/home/anthony/sonar-processing/bvtsdk/colormaps/bone.cmap");
      if( ret != 0 )
      {
 	  printf("BVTColorMapper_Load: ret=%d\n", ret);
 	  return 1;
      }

      //cout << "AUTOTHRESH = " << BVTColorMapper_GetAutoMode(mapper) << endl;
      BVTColorMapper_SetAutoMode(mapper, 0);
      //cout << "AUTOTHRESH = " << BVTColorMapper_GetAutoMode(mapper) << endl;
      //cout << "Top Threshold = " << BVTColorMapper_GetTopThreshold(mapper) << endl;
      //cout << "Bottom Threshold = " << BVTColorMapper_GetBottomThreshold(mapper) << endl;
	
 	//Create Windows:
      	char color_wnd[] = "Sonar: 'b'=back, 'f'=forward, 'p'=pause, 'ESC'=exit";
      	cvNamedWindow(color_wnd, 1);
 	cv::Point pt1,pt2;
 	double theta;

	//Create Trackbars:
	//char position_tbar[] = "Position";
	//cv::createTrackbar(position_tbar, video_wnd, &i, max_frame_num, Trackbar);

	char text[255];

	int threshold = 0;
	int athreshold = -7;
	int athresholdblksz = 2;

	//char thresh_trackbar[] = "Threshold Value";
	//cv::createTrackbar(thresh_trackbar,thresh_wnd,&threshold,255,NULL);
	//char adpt_thresh_trackbar[] = "Adaptive Threshold Value";
	//cv::createTrackbar(adpt_thresh_trackbar,adpt_thresh_wnd,&athreshold,10,NULL);
	//char adpt_blk_trackbar[] = "Adaptive Threshold Block";
	//cv::createTrackbar(adpt_blk_trackbar,adpt_thresh_wnd,&athresholdblksz,10,NULL);

	cv::Mat sonarImgGray, binary_img, morph_img, components_img;

	//Create Trackbars:             
	int max_frame_num = pings - 1;
        char position_tbar[] = "Position (0 based)";
	cv::createTrackbar(position_tbar, color_wnd, &i, max_frame_num, Trackbar);

	//---------------------------------------------------------------------
	//----------------Main Loop--------------------------------------------

	//pings = 300;//DEBUG

     for (i = 0; i < pings; i++) 
       {
	 
	 BVTMagImage bvtmag_img;	//Magnitude Image
	 BVTColorImage bvtclr_img;	//Color Mapped Image      
	 
	 //Create Color Image
	 IplImage* color_img_ipl;
	 
	 //CREATE GRAY IMAGE
	 IplImage* mag_img_ipl;
	 
	 //	 cout << i << "/" << pings << endl;

	  // Now, get a ping!
	  BVTPing ping = NULL;
	  //if ( i == 0) {
	  ret = BVTHead_GetPing(head, i, &ping);
	  //} else {
	  //ret = BVTHead_GetPing(head, -1, &ping);
	  //}
	  if( ret != 0 )
	  {
	       printf("BVTHead_GetPing: ret=%d\n", ret);
	       return 1;
	  }

	  //Get Ping0 Timestamp for TIME0
	  if(i == 0)
	    SONAR_TIME0 = BVTPing_GetTimestamp(ping);

	  // Generate an image from the ping	  
	  //ret = BVTPing_GetImage(ping, &bvtmag_img);
	  //ret = BVTPing_GetImageXY(ping, &bvtmag_img);
	  ret = BVTPing_GetImageRTheta(ping, &bvtmag_img);
	  if( ret != 0 )
	  {
	       printf("BVTPing_GetImage: ret=%d\n", ret);
	       return 1;
	       continue;
	  }

	  //Get Ping Timestamp and find time from Ping0
	  ping_time_sec = BVTPing_GetTimestamp(ping) - SONAR_TIME0;

	  std::cout << i <<  "/" << pings <<  " , " << ping_time_sec << std::endl;

	  height_mag = BVTMagImage_GetHeight(bvtmag_img);
	  width_mag = BVTMagImage_GetWidth(bvtmag_img);
	  if((height_mag==0)||(width_mag==0)) //If empty frame
	    {
	      cout << "Empty frame...continuing" << endl;
	      continue;
	    }
	  double rangeResMag = BVTMagImage_GetRangeResolution(bvtmag_img);
          double bearingResMag = BVTMagImage_GetBearingResolution(bvtmag_img);
	  min_bearingRes = min(bearingResMag,min_bearingRes);
	  max_bearingRes = max(bearingResMag,max_bearingRes);
          //cout << "Range Resolution MAG " << rangeResMag << " meters/pixel" << endl;
	  //         cout << "Bearing Resolution MAG " << bearingResMag << " deg/pixel" << endl;
	  
	  mag_img_ipl = cvCreateImageHeader(cvSize(width_mag,height_mag), IPL_DEPTH_8U, 1);
	  cvSetImageData(mag_img_ipl, BVTMagImage_GetBits(bvtmag_img), width_mag);
      
	  /////////////////////////////////////////////////////////
	
	  // Perform the colormapping
	  ret = BVTColorMapper_MapImage(mapper, bvtmag_img, &bvtclr_img);
	  if( ret != 0 )
	  {
	       printf("BVTColorMapper_MapImage: ret=%d\n", ret);
	       return 1;
	  }
	
	  /////////////////////////////////////////////////////////
	  // Use OpenCV to display the image
	  height = BVTColorImage_GetHeight(bvtclr_img);
	  width = BVTColorImage_GetWidth(bvtclr_img);
	  double rangeRes = BVTColorImage_GetRangeResolution(bvtclr_img);
          //cout << "Range Resolution COLOR " << rangeRes << " meters/pixel" << endl;
	  min_rangeRes = min(rangeRes,min_rangeRes);
	  max_rangeRes = max(rangeRes,max_rangeRes);
	  min_height = min((double)height,min_height);
	  max_height = max((double)height,max_height);
	  min_width = min((double)width,min_width);
	  max_width = max((double)width,max_width);
	  //cout << "Min (rangeres,bearingres,height,width)=" << min_rangeRes << "," << min_height << "," << min_width << endl;
	  //cout << "Max (rangeres,bearingres,height,width)=" << max_rangeRes << "," << max_height << "," << max_width << endl;

	  // Create a IplImage header
	  color_img_ipl = cvCreateImageHeader(cvSize(width,height), IPL_DEPTH_8U, 4);
	
	  // And set it's data
	  cvSetImageData(color_img_ipl,  BVTColorImage_GetBits(bvtclr_img), width*4);

	  //Convert IPL image to Mat:
	  //cv::Mat colorImg(color_img_ipl);
	  Mat colorImg = cvarrToMat(color_img_ipl,true);

	  //Delete ipl image header and bvt imgs                                 
	  BVTPing_Destroy(ping);
	  cvReleaseImageHeader(&color_img_ipl);
	  cvReleaseImageHeader(&mag_img_ipl);
	  BVTColorImage_Destroy(bvtclr_img);
	  BVTMagImage_Destroy(bvtmag_img);

	if(WRITE_FRAME_NUM){
	  //Put Frame Number on image
	  sprintf(text, "%d", i);
	  cv::putText(colorImg, text, cvPoint(0,25), cv::FONT_HERSHEY_SIMPLEX,1,cv::Scalar::all(255));
	}
	
 
	//-------------------------------------------

	//Display Images:
	cv::imshow(color_wnd, colorImg);

	if(VERBOSE&&(i==1))
		printf("height: %d width: %d\n", height, width);

	  outFile << i << ";" << rangeResMag << ";" << bearingResMag << ";" << rangeRes << ";" << height << ";" << width << ";" << endl;

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
		else if(keypushed=='s')
		  {
		    imwrite("output.png",sonarImgGray);
		  }
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

     cout << "Min (rangeres,bearingres,height,width)=" << min_rangeRes << "," << min_bearingRes << "," << min_height << "," << min_width << endl;
     cout << "Max (rangeres,bearingres,height,width)=" << max_rangeRes << "," << max_bearingRes << "," << max_height << "," << max_width << endl;	
     cout << endl;
     //cvReleaseImageHeader(&color_img_ipl);
     //cvReleaseImageHeader(&mag_img_ipl);

     // Clean up
     //BVTColorImage_Destroy(bvtclr_img);
     //BVTMagImage_Destroy(bvtmag_img);
     BVTColorMapper_Destroy(mapper);
     BVTSonar_Destroy(son);

     outFile.close();

     return 0;
}
