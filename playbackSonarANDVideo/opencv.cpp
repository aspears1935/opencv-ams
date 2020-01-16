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
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include <highgui.h>
#include <cmath>

 using namespace std;
 using namespace cv;

 #define SAVE_IMAGES 	false
 #define GET_RAW_DATA 	false
 #define GRID_ON		false
 #define WRITE_FRAME_NUM true
// #define SYNCH_OFFSET_SEC -0.77411 //Offset between sonar and video data (positive = video leading)
//DEBUG::::
#define SYNCH_OFFSET_SEC 0 //3194 //1170 //884 //132 //-583 //32 //1869 //1421 //953 //474 //-6 //1566 //661 //54//693 //33//34 //Offset between sonar and video data (positive = video leading) //33 for 
//END DEBUG
 #define DISPLAY_COMPASS_DEPTH false
 #define PI 3.14159265359
 #define DEPTH_OFFSET 60
 #define max_BINARY_value 255
 #define SON_FOV_X 45
 #define VID_FOV_X 85
 #define VERBOSE true
 #define REMAP false
 #define DRAW_BOXES true
 #define WRITE_VIDEO true

//double camArr30degWin[3][3] = {471.272, 0, 359.5,0, 468.339, 269.5, 0,0,1};
//double camArr30degWin[3][3] = {471.272, 0, 359.5,0, 468.339, 239.5, 0,0,1}; //Run11
//double distArr30degWin[5][1] = {-0.318162, 0.137179, 0.00494844, 0.00176499, -0.040086};

double camArr0degWin[3][3] = {450.247, 0, 359.5,0, 452.456, 269.5, 0,0,1};
double distArr0degWin[5][1] = {-0.297356, 0.11297, 0.00309039, 0.00158565, -0.0248698};

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
 	double frame_num_windows = 0; //Frame Number for Video in Windows data
 	int frame_num_windows_int = 0; // Same as an int rounded
 	double heading = 0;
 	double heading_deg = 0;
 	double depth = 0;
 	int SON_PIX_X;
 	int VID_PIX_X;

      if (argc < 3) {
 	  printf("usage: ./sonar_opencv <sonar-file> <video-file> <OPTIONAL compass-depth-file>\n");
 	  printf("example: ./sonar_opencv ../../../data/swimmer.son ../../../data/swimmer.avi ../../../data/compasssonar.csv\n");
 	  exit(-1);
      }

      char DataFile[256];
      char videoFileName[256];
      char CompassFileName[256];
      strcpy(DataFile, argv[1]);
      strcpy(videoFileName, argv[2]);
      if(argc > 3)
        strcpy(CompassFileName, argv[3]);

      //Undistortion Initialization
      Mat cameraMatrix = Mat(3,3,DataType<double>::type,camArr0degWin);
      Mat distCoeffs = Mat(5,1,DataType<double>::type,distArr0degWin);
      Mat newCameraMatrix = cameraMatrix.clone();
      Size videoSizeTmp(720,540);
      double fx_new = (videoSizeTmp.width/2)/(tan((CV_PI/180)*((double)VID_FOV_X)/2));
      double fx_old = cameraMatrix.at<double>(0,0);
      double fy_old = cameraMatrix.at<double>(1,1);
      double fy_new = (fx_new*fy_old)/fx_old;
      Point2d princ_pt = Point2d(cameraMatrix.at<double>(0,2),cameraMatrix.at<double>(1,2));
      newCameraMatrix.at<double>(0,0) = fx_new;
      newCameraMatrix.at<double>(1,1) = fy_new;
      Mat map1, map2;
      initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(), newCameraMatrix, videoSizeTmp, CV_32FC1, map1, map2);
      cout << newCameraMatrix << endl;

  
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
      printf("BVTHead_GetPingCount: %d\n", pings);

      // Set the range window to be 10m to 40m
      BVTHead_SetRange(head, 0, 50);

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

      cout << "AUTOTHRESH = " << BVTColorMapper_GetAutoMode(mapper) << endl;
      BVTColorMapper_SetAutoMode(mapper, 0);
      cout << "AUTOTHRESH = " << BVTColorMapper_GetAutoMode(mapper) << endl;
      cout << "Top Threshold = " << BVTColorMapper_GetTopThreshold(mapper) << endl;
      cout << "Bottom Threshold = " << BVTColorMapper_GetBottomThreshold(mapper) << endl;
 	//Create Color Image
      	IplImage* color_img_ipl;

 	//CREATE GRAY IMAGE
      	IplImage* mag_img_ipl;
	
 	//Create Windows:
      	char color_wnd[] = "Sonar";
      	cvNamedWindow(color_wnd, 1);
 	char video_wnd[] = "Video: 'b'=back, 'f'=forward, 'p'=pause, 'ESC'=exit";
 	cvNamedWindow(video_wnd,1);
 	char compass_wnd[] = "Compass";
 	char depth_wnd[] = "Depth";
 	char sonar_box_wnd[] = "Sonar Boxes";
 	cvNamedWindow(sonar_box_wnd,1);
 	cv::Mat compassImg;
 	cv::Mat depthImg;
 	cv::Point pt1,pt2;
 	double theta;
 	moveWindow(video_wnd,400,0);
 	moveWindow(sonar_box_wnd,1000,0);
       	if(DISPLAY_COMPASS_DEPTH)
	  {
	    cvNamedWindow(compass_wnd,1);
	    cvNamedWindow(depth_wnd,1);
	    compassImg = cv::Mat::zeros(200,200,CV_8UC3);
	    depthImg = cv::Mat::zeros(300,50,CV_8UC3);
	  }

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

	//--------------------------------------------------
	cv::VideoCapture inputVideo(videoFileName);
	if(!inputVideo.isOpened())
	  {
	    std::cout << "Could not open the input video: " << videoFileName << std::endl;
	    return -1;
	  }

	cv::Size videoSize = cv::Size((int) inputVideo.get(CV_CAP_PROP_FRAME_WIDTH),    // Acquire input size
		      (int) inputVideo.get(CV_CAP_PROP_FRAME_HEIGHT));

	std::cout << "Input frame resolution: Width=" << videoSize.width << "  Height=" << videoSize.height << " Num of Frames: " << inputVideo.get(CV_CAP_PROP_FRAME_COUNT) << " FPS=" << inputVideo.get(CV_CAP_PROP_FPS) << std::endl;
	cv::Mat videoImg;
	cv::Mat sonarImgGray, binary_img, morph_img, components_img;
	double VIDEO_FPS = inputVideo.get(CV_CAP_PROP_FPS);
	double VIDEO_FRAME_COUNT = inputVideo.get(CV_CAP_PROP_FRAME_COUNT);
	VID_PIX_X = videoSize.width;
	cout << "VIDEO_FPS = " << VIDEO_FPS << endl;
	cout << "VID_PIX_X = " << VID_PIX_X << endl;

	//Create Trackbars:             
	int max_frame_num = pings - 1;
        char position_tbar[] = "Position (0 based)";
	cv::createTrackbar(position_tbar, video_wnd, &i, max_frame_num, Trackbar);

	//Check if Windows Data (More frames in video than sonar)
	if(VIDEO_FRAME_COUNT > pings)
	  {
	    Windows_Data = true;
	    cout << "WINDOWS DATA" << endl;
	  }

	//--------------------------------------------------------------------
	//------------------Compass And Depth Data----------------------------
	    ifstream inFileCompass(CompassFileName);
	    string tmpstring;
	if(argc > 3)
	  {
	    getline(inFileCompass,tmpstring,'\n');
	    cout << tmpstring << endl;
	  }

	//Create Array for Heading/Depth data
	float * heading_arr;
	float * depth_arr;
	heading_arr = new float[pings];
	depth_arr = new float[pings];

	//-------------------------------------------------------
	//-------------Read in compass, depth readings
	for(i=0; i<pings; i++)
	{
	  getline(inFileCompass,tmpstring,','); //Ignore timestamp
	  getline(inFileCompass,tmpstring,',');
	  heading_arr[i] = (float)(atof(tmpstring.c_str()));
	  getline(inFileCompass,tmpstring,'\n');
	  depth_arr[i] = (float)(atof(tmpstring.c_str()));
	}

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

	//-------------Create Video Writer-----------------
	VideoWriter outputVideo;
	int outputFPS = 10;
	if(WRITE_VIDEO)
	  {
	    outputVideo.open("output.avi",cv::VideoWriter::fourcc('X','V','I','D'), outputFPS, Size(videoSize.width*2,videoSize.height),true);
	    if(!outputVideo.isOpened())
	      {
		cout << "Could not open the output video for write" << endl;
		return -1;
	      }
	  }

	//---------------------------------------------------------------------
	//----------------Main Loop--------------------------------------------

	//pings = 300;//DEBUG

     for (i = 0; i < pings; i++) 
       {
 	//Create Color Image
      	IplImage* color_img_ipl;

 	//CREATE GRAY IMAGE
      	IplImage* mag_img_ipl;

	BVTMagImage bvtmag_img;	//Magnitude Image
	BVTColorImage bvtclr_img;	//Color Mapped Image

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
	  ret = BVTPing_GetImageXY(ping, &bvtmag_img);
	  //ret = BVTPing_GetImageRTheta(ping, &bvtmag_img);
	  if( ret != 0 )
	  {
	       printf("BVTPing_GetImage: ret=%d\n", ret);
	       return 1;
	  }

	  height_mag = BVTMagImage_GetHeight(bvtmag_img);
	  width_mag = BVTMagImage_GetWidth(bvtmag_img);
	

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
	  SON_PIX_X = width;
	  if(VERBOSE)
	    cout << "---------------------" << endl << "SON_PIX_X = " << SON_PIX_X << endl;
	  
	  // Create a IplImage header
	  color_img_ipl = cvCreateImageHeader(cvSize(width,height), IPL_DEPTH_8U, 4);
	
	  // And set it's data
	  cvSetImageData(color_img_ipl,  BVTColorImage_GetBits(bvtclr_img), width*4);

	  //Convert IPL image to Mat:
	  //cv::Mat colorImg(color_img_ipl);
	  Mat colorImg = cvarrToMat(color_img_ipl,true);

	if(WRITE_FRAME_NUM){
	  //Put Frame Number on image
	  sprintf(text, "%d", i);
	  cv::putText(colorImg, text, cvPoint(0,25), cv::FONT_HERSHEY_SIMPLEX,1,cv::Scalar::all(255));
	}
	
	//Get Ping Timestamp and find time from Ping0
	ping_time_sec = BVTPing_GetTimestamp(ping) - SONAR_TIME0;
	frame_num_windows = (ping_time_sec + SYNCH_OFFSET_SEC)*VIDEO_FPS;
	frame_num_windows_int = (int)round(frame_num_windows);

	if(frame_num_windows_int < 0) //Can't have negative frame num
	  frame_num_windows_int = 0;
	else if(frame_num_windows_int > (VIDEO_FRAME_COUNT-1))
	  frame_num_windows_int = (int)(VIDEO_FRAME_COUNT-1);

	std::cout << i <<  "/" << pings << "==" << frame_num_windows_int << "/" << VIDEO_FRAME_COUNT <<  ", " << ping_time_sec << "===" << frame_num_windows_int/VIDEO_FPS << std::endl;

	BVTPing_Destroy(ping);
	cvReleaseImageHeader(&color_img_ipl);
	cvReleaseImageHeader(&mag_img_ipl);
	BVTColorImage_Destroy(bvtclr_img);
	BVTMagImage_Destroy(bvtmag_img);
 
	//-------------------------------------------
	inputVideo.set(CV_CAP_PROP_POS_FRAMES,i);
	if(Windows_Data) //IF WINDOWS DATA, reset the position to calculated frame
	  {
	    inputVideo.set(CV_CAP_PROP_POS_FRAMES, frame_num_windows_int);
	    //cout << "Got Windows Video Frame " << frame_num_windows_int << endl; //DEBUG DELETE!
	  }
	else
	  cout << "Got Video Frame " << i << "--" << frame_num_windows_int << endl; //DEBUG DELETE!
	inputVideo >> videoImg;
	//	cout << "Windows Frame " << frame_num_windows_int << " Empty = " << videoImg.empty() << " - get(POS_FRAMES)=" << inputVideo.get(CV_CAP_PROP_POS_FRAMES) << endl;

	//std::cout << inputVideo.get(CV_CAP_PROP_POS_MSEC) << std::endl;

	//FIX FOR BAD FRAMES AT END OF WINDOWS DATA
	/*while(videoImg.empty())
	  {
	    cout << "HIT EMPTY VIDEO FRAME!" << frame_num_windows_int << endl;
	    frame_num_windows_int--;
	    //frame_num_windows_int++;
	    inputVideo.set(CV_CAP_PROP_POS_FRAMES, frame_num_windows_int);
	    cout << "get(POS_FRAMES)=" << inputVideo.get(CV_CAP_PROP_POS_FRAMES) << endl;
	    inputVideo >> videoImg;
	    cout << "Frame " << frame_num_windows_int << " Empty = " << videoImg.empty() << endl;
	    }*/
	if(videoImg.empty())
	  videoImg = cv::Mat::zeros(videoSize.height,videoSize.width,CV_8UC3);

	if(REMAP)
	  remap(videoImg,videoImg,map1,map2,INTER_LINEAR);

	//-----------------------------------------------------
	//-------------- Get Sonar Blobs ------------------------
	cvtColor(colorImg, sonarImgGray, COLOR_BGR2GRAY);
	//equalizeHist(sonarImgGray, sonarImgGray, max_BINARY_value, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 3, 0);
	//Threshold:
	//int thresh_level = 100;
      	cv::threshold(sonarImgGray, binary_img, 100, max_BINARY_value, 3);
	imwrite("sonarGray.png",sonarImgGray);
	imwrite("sonarThresh.png",binary_img);
	//Morphology:
       	Mat element = getStructuringElement(MORPH_RECT,Size(3,3),Point(1,1));
	erode(binary_img, morph_img, element,Point(-1,-1),2);
	dilate(morph_img, morph_img, element,Point(-1,-1),2);
	imwrite("openedImg.png",morph_img);
		
	//Components:
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(morph_img,contours,hierarchy,CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE);
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	vector<Point2f> center(contours.size());
	vector<float> radius(contours.size());
	Mat sonImgBoxes = sonarImgGray.clone();
	
	if(DRAW_BOXES)
	  {
	    for(int count = 0; count < contours.size(); count++)
	      {
		approxPolyDP(Mat(contours[count]),contours_poly[count], 3, true);
		boundRect[count] = boundingRect(Mat(contours_poly[count]));
		minEnclosingCircle((Mat)contours_poly[count], center[count], radius[count]);
		Scalar color = Scalar(rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255));
		drawContours(sonImgBoxes, contours_poly, count, color, 1, 8, vector<Vec4i>(),0,Point());
		imwrite("sonarConnectedComp.png",sonImgBoxes);
		rectangle(sonImgBoxes, boundRect[count].tl(), boundRect[count].br(), color, 2, 8, 0);
		
		//Draw Sonar Lines On Video Images:
		//float xpointtemp = ((center[count].x - (SON_PIX_X/2))*(SON_FOV_X/SON_PIX_X)*(VID_PIX_X/VID_FOV_X)) + (VID_PIX_X/2);
		double temporary;
		temporary = center[count].x - (SON_PIX_X/2);
		temporary = temporary*SON_FOV_X;
		temporary = temporary/SON_PIX_X;
		temporary = temporary*VID_PIX_X;
		temporary = temporary/VID_FOV_X;
		temporary = temporary + (VID_PIX_X/2);
		
		/*pt1.x = temporary;
		  pt1.y = 0;
		  pt2.x = temporary;
		  pt2.y = 480;
		  line(videoImg,pt1,pt2,Scalar(0,0,0),5,8); */
		//	    pt1.x = temporary - 100;
		double boxRadius =  ((((radius[count]*SON_FOV_X)/SON_PIX_X)*VID_PIX_X)/VID_FOV_X)*10;
		cout << boxRadius << endl;
		pt1.x = temporary - boxRadius;
		pt1.y = 50;
		pt2.x = temporary + boxRadius;
		pt2.y = 480 - 50;
		rectangle(videoImg, pt1, pt2, Scalar(0,0,255),5,8); 
	      }
	  }

	imshow(sonar_box_wnd, sonImgBoxes);
	imwrite("sonarBoxes.png",sonImgBoxes);

	//-------------------------------------------------------
	//-------------Read in compass, depth readings
      	heading_deg = heading_arr[i];
	depth = depth_arr[i];

	if(DISPLAY_COMPASS_DEPTH)
	  cout << "Heading = " << heading_deg << "  Depth = " << depth << endl;

	if(WRITE_FRAME_NUM){
	  //Put Frame Number on image                                               
	  if(Windows_Data)
	    sprintf(text, "%d", frame_num_windows_int);
	  else //Else it is Linux data with 1:1 frame ratio
	    sprintf(text, "%d", i);
	  cv::putText(videoImg, text, cvPoint(0,25), cv::FONT_HERSHEY_SIMPLEX,1,cv::Scalar::all(255));
        }

	if(DISPLAY_COMPASS_DEPTH)
	  {
	    //DRAW COMPASS:
	    compassImg = cv::Mat::zeros(200,200,CV_8UC3);
	    //circle(img,center,radius,color,thickness,lineType,shift);
	    cv::circle(compassImg,cv::Point(100,100),90,cv::Scalar(255,255,255),5,8,0);
	    heading = heading_deg*PI/180;
	    pt1.x = 100; pt1.y=100; //Center
	    pt2.x = 100 + 90*sin(heading);
	    pt2.y = 100 - 90*cos(heading);
	    cv::line(compassImg, pt1, pt2, cv::Scalar(255,255,255),5,8);

	    //DRAW DEPTH:
	    depthImg = Mat::zeros(300,50,CV_8UC3);
	    //depth = 100;
	    depth = depth - DEPTH_OFFSET;
	    pt1.x = 0;
	    pt1.y = depth*5;
	    pt2.x = 50;
	    pt2.y = depth*5;
	    cv::line(depthImg, pt1, pt2, cv::Scalar(255,255,255),5,8);
	  }


	//Display Images:
	cv::imshow(color_wnd, colorImg);
	//std::cout << frame_num_windows_int << std::endl;
	cv::imshow(video_wnd, videoImg);
	if(DISPLAY_COMPASS_DEPTH)
	  {
	    cv::imshow(compass_wnd, compassImg);
	    cv::imshow(depth_wnd, depthImg);
	  }
	//Write out video:
	//outputSonar.write(colorImg);
	//outputVideo.write(videoImg);

	//Write Video Out----------------:
	if(WRITE_VIDEO)
          {
            //Write output OF video frame--------------------------------------- 
	    Mat outVideoImg = Mat::zeros(videoImg.rows,videoImg.cols*2,CV_8UC3);
            Mat tmp1, tmp2, tmp3, tmp4, tmp5;

	    cvtColor(sonImgBoxes,tmp5,COLOR_GRAY2BGR);
            resize(tmp5, tmp1, Size(videoSize.width,videoSize.height));
            Mat mapRoi1(outVideoImg, Rect(0, 0, videoImg.cols, videoImg.rows));
            tmp1.copyTo(mapRoi1);

	    //	    cvtColor(videoImgUndistort,tmp5,COLOR_GRAY2BGR);
            resize(videoImg, tmp2, Size(videoSize.width,videoSize.height));
            Mat mapRoi2(outVideoImg, Rect(videoImg.cols, 0, videoImg.cols, videoImg.rows));
            tmp2.copyTo(mapRoi2);

            outputVideo << outVideoImg;
	  }


	//print frame information	
	//printf("i=%d\n", i);
	if(i==1)
		printf("height: %d width: %d\n", height, width);


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
	
     //cvReleaseImageHeader(&color_img_ipl);
     //cvReleaseImageHeader(&mag_img_ipl);

     cvDestroyWindow(color_wnd);
     cvDestroyWindow(video_wnd);
     cvDestroyWindow(compass_wnd);
     cvDestroyWindow(depth_wnd);
	
     // Clean up
     //BVTColorImage_Destroy(bvtclr_img);
     //BVTMagImage_Destroy(bvtmag_img);
     BVTColorMapper_Destroy(mapper);
     BVTSonar_Destroy(son);

     inFileCompass.close();

     return 0;
}
