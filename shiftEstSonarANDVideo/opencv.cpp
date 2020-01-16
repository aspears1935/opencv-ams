/*
To Do:
- Implement 3DOF rotation,translation for sonar OR affine transform
- Implement MAPSAC


*/

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
#include <cmath>

#include <bvt_sdk.h>

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
#include "RANSAC.hpp"
#include "my_ptsetreg.hpp"
#include "my_five-point.hpp"

using namespace std;
using namespace cv;

#define LAPTOP_CAMERA false //Also need to change DES_FOVX and other FOVX
#define DISPLAY_IMGS 	true
#define DISPLAY_POSE    false
#define GET_RAW_DATA 	false
#define GRID_ON		false
#define WRITE_FRAME_NUM true
#define BLUR_SON_IMGS   true
#define BLUR_VID_IMGS   true
#define THRESHOLD_SON_IMGS false
#define SONAR_CARTESIAN true

#define SYNCH_OFFSET_SEC -0.77411 //Offset between sonar and video data (video leading)
#define DISPLAY_COMPASS_DEPTH true
#define PI 3.14159265359
#define DEPTH_OFFSET 60
#define max_BINARY_value 255
#define DES_FOVX 80 //Desired FOV in the X direction
//#define DES_FOVX 50 //DEBUG
#define LINUX0DEG  0
#define LINUX17DEG 1
#define LINUX30DEG 2
#define LINUX68DEG 3
#define WINDOWS0DEG 4
#define WINDOWS17DEG 5
#define WINDOWS30DEG 6
#define WINDOWS68DEG 7
#define SON_FOV_X 45
#define VID_FOV_X 80 //FOV in X direction in video data 
//#define VID_FOV_X 50 //DEBUG
#define MAPSAC 7
#define ROBUST_EST_METH CV_FM_LMEDS //CV_FM_LMEDS or CV_FM_RANSAC or MAPSAC

//Initialize the camera matrices
double camArr0degLinux[3][3] = {312.613, 0, 247.5,0, 413.636, 239.5, 0,0,1};
double camArr17degLinux[3][3] = {319.202, 0, 247,0, 421.647, 239.5, 0,0,1};
double camArr30degLinux[3][3] = {326.638, 0, 247.5,0, 432.628, 239.5, 0,0,1};
//double camArr30degLinux[3][3] = {432.628, 0, 359.5,0, 432.628, 287.5, 0,0,1}; DEBUG
double camArr68degLinux[3][3] = {316.409, 0, 247.5,0, 415.958, 239.5, 0,0,1};
double camArr0degWin[3][3] = {450.247, 0, 359.5,0, 452.456, 269.5, 0,0,1};
double camArr17degWin[3][3] = {474.259, 0, 359.5,0, 474.039, 269.5, 0,0,1};
double camArr30degWin[3][3] = {471.272, 0, 359.5,0, 468.339, 269.5, 0,0,1};
double camArr68degWin[3][3] = {471.59, 0, 359.5,0, 477.484, 269.5, 0,0,1};
double camArrLaptop[3][3] = {677.694, 0, 319.5, 0, 676.564, 239.5, 0, 0, 1};
//Initialize the Distortion matrices
double distArr0degLinux[5][1] = 
  {-0.310408, 0.135956, 0.00476035, 0.0011647, -0.0406918};
double distArr17degLinux[5][1] = 
  {-0.319807, 0.139937, 0.00323336, 0.00140327, -0.0377739};
double distArr30degLinux[5][1] = 
  {-0.332741, 0.143602, 0.00376436, -0.00288967, -0.0381608};
double distArr68degLinux[5][1] = 
  {-0.293871, 0.080991, 0.000844937, -0.00066441, -0.00787792};
double distArr0degWin[5][1] = 
  {-0.297356, 0.11297, 0.00309039, 0.00158565, -0.0248698};
double distArr17degWin[5][1] = 
  {-0.321047, 0.125535, 0.00217876, 0.00194057, -0.0306027};
double distArr30degWin[5][1] = 
  {-0.318162, 0.137179, 0.00494844, 0.00176499, -0.040086};
double distArr68degWin[5][1] = 
  {-0.319671, 0.128974, 0.00161037, 0.0000403, -0.0273568};
double distArrLaptop[5][1] = 
  {0.0582694, -0.391214, 0.016215, -0.00911916, 0.52075};
int camDataType = 0; //Windows or Linux and angle of sensor


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


   //testEstimateTranslationNew();   
   //testEstimateRigidTransform2DNew();   
   //rigidTransform2D(10); //DEBUG - test this function
   //for(;;);

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
	//Initialize the sum angles
	Mat vidOFsumAngles = Mat::zeros(3,1,DataType<double>::type);
	Mat vidSURFsumAngles = Mat::zeros(3,1,DataType<double>::type);
	Mat vidSIFTsumAngles = Mat::zeros(3,1,DataType<double>::type);
	Mat vidHARRISsumAngles = Mat::zeros(3,1,DataType<double>::type);

	
	//Print out OPENCV Version:
	cout << "Using OpenCV v" << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << "." << CV_SUBMINOR_VERSION << endl;

      if (argc < 3) {
 	  printf("usage: ./sonar_opencv <sonar-file> <video-file> <OPTIONAL compass-depth-file> <OPTIONAL sensor angle>\n");
 	  printf("example: ./sonar_opencv ../../../data/swimmer.son ../../../data/swimmer.avi ../../../data/compasssonar.csv 30\n");
 	  exit(-1);
      }

      char DataFile[256];
      char videoFileName[256];
      char CompassFileName[256];
      int sensorAngle;
      strcpy(DataFile, argv[1]);
      strcpy(videoFileName, argv[2]);
      if(argc > 3)
        strcpy(CompassFileName, argv[3]);
      if(argc > 4)
	  sensorAngle = (int)argv[4];

      //Open Output Files:
      ofstream outfile("output.csv");
      outfile << "frame1;frame2;compass;compassChange;vidOF_x;vidOF_y;vidOF_z;vidOF_x_rot;vidOF_y_rot;vidOF_z_rot;vidOF_x_sum;vidOF_y_sum;vidOF_z_sum;";
      outfile << "vidSURF_x;vidSURF_y;vidSURF_z;vidSURF_x_rot;vidSURF_y_rot;vidSURF_z_rot;vidSURF_x_sum;vidSURF_y_sum;vidSURF_z_sum;";
      outfile << "vidSIFT_x;vidSIFT_y;vidSIFT_z;vidSIFT_x_rot;vidSIFT_y_rot;vidSIFT_z_rot;vidSIFT_x_sum;vidSIFT_y_sum;vidSIFT_z_sum;";
      outfile << "vidHARRIS_x;vidHARRIS_y;vidHARRIS_z;vidHARRIS_x_rot;vidHARRIS_y_rot;vidHARRIS_z_rot;vidHARRIS_x_sum;vidHARRIS_y_sum;vidHARRIS_z_sum;" << endl;
  
      ofstream outfileSonar("outputSonar.csv");
      outfileSonar << "frame1;frame2;compass;compassChange;sonOFxshift;sonOFyshift;sonOFtheta;sonSURFxshift;sonSURFyshift;sonSURFtheta;sonSIFTxshift;sonSIFTyshift;sonSIFTtheta;sonHARRISxshift;sonHARRISyshift;sonHARRIStheta" << endl;

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

      BVTMagImage bvtmag_img;	//Magnitude Image
      BVTColorImage bvtclr_img;	//Color Mapped Image

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
      //Mat color_img_ipl;

 	//CREATE GRAY IMAGE
      IplImage* mag_img_ipl;
      //Mat mag_img_ipl;

 	//Create Windows:
      	char color_wnd[] = "Sonar";
      	namedWindow(color_wnd, 1);
 	char video_wnd[] = "Video: 'b'=back, 'f'=forward, 'p'=pause, 'ESC'=exit";
 	namedWindow(video_wnd,1);
	char video_undist_wnd[] = "Undistorted Video";
	namedWindow(video_undist_wnd,1);
 	char compass_wnd[] = "Compass";
 	char depth_wnd[] = "Depth";
 	char sonar_box_wnd[] = "Sonar Boxes";
	char son_features_wnd[] = "Sonar Features Window";
	char son_matches_wnd[] = "Sonar Matches Window";
	char vid_features_wnd[] = "Video Features Window";
	char vid_matches_wnd[] = "Video Matches Window";
	
 	namedWindow(sonar_box_wnd,1);
	namedWindow(vid_features_wnd,1);
 	cv::Mat compassImg;
 	cv::Mat depthImg;
 	cv::Point pt1,pt2;
 	double theta;
 	moveWindow(video_wnd,400,0);
 	moveWindow(sonar_box_wnd,1000,0);
       	if(DISPLAY_COMPASS_DEPTH)
	  {
	    namedWindow(compass_wnd,1);
	    namedWindow(depth_wnd,1);
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

	cv::Size videoSize = cv::Size((int) inputVideo.get(CAP_PROP_FRAME_WIDTH),    // Acquire input size
		      (int) inputVideo.get(CAP_PROP_FRAME_HEIGHT));
	int numVidFrames = inputVideo.get(CAP_PROP_FRAME_COUNT);
	std::cout << "Input frame resolution: Width=" << videoSize.width << "  Height=" << videoSize.height << " Num of Frames: " << numVidFrames << " FPS=" << inputVideo.get(CAP_PROP_FPS) << std::endl;
	cv::Mat videoImg;
	cv::Mat sonarImgGray, binary_img, morph_img, components_img, vidFeaturesImg,sonFeaturesImg, vidMatchesImg, sonMatchesImg;
	double VIDEO_FPS = inputVideo.get(CAP_PROP_FPS);
	double VIDEO_FRAME_COUNT = inputVideo.get(CAP_PROP_FRAME_COUNT);
	VID_PIX_X = videoSize.width;
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


	//-----------------------------------------------------------
	//------------------- Initialize Feature Detection ----------
	//-----------------------------------------------------------
	//SURF Detector:
	int minHessian = 300; //Suggested 300-500. Higher val gives less features.
	SurfFeatureDetector sonSURFdetector(minHessian);
	SurfDescriptorExtractor sonSURFextractor;
	FlannBasedMatcher sonSURFmatcher;
	SurfFeatureDetector vidSURFdetector(minHessian);
	SurfDescriptorExtractor vidSURFextractor;
	FlannBasedMatcher vidSURFmatcher;

	//SIFT Detector:
	cv::Ptr< cv::FeatureDetector > sonSIFTdetector = FeatureDetector::create("SIFT");
	cv::Ptr< cv::DescriptorExtractor > sonSIFTextractor = DescriptorExtractor::create("SIFT");
	FlannBasedMatcher sonSIFTmatcher;
	
	cv::Ptr< cv::FeatureDetector > vidSIFTdetector = FeatureDetector::create("SIFT");
	cv::Ptr< cv::DescriptorExtractor > vidSIFTextractor = DescriptorExtractor::create("SIFT");
	double SIFTthreshold = 0.001; //0.04
	double SIFTedgeThreshold = 100; //10 -> larger = more features
	vidSIFTdetector->set("contrastThreshold", SIFTthreshold);
	vidSIFTdetector->set("edgeThreshold", SIFTedgeThreshold);
	vidSIFTdetector->set("nFeatures", 100);
	vidSIFTdetector->set("nOctaveLayers", 5);
	vidSIFTdetector->set("sigma", 1.0);//1.6 lower seems to be better to a point
	FlannBasedMatcher vidSIFTmatcher;
	//BFMatcher vidSIFTmatcher;

	//DEBUG:
	/*cv::Ptr< cv::FeatureDetector > vidSIFTdetector1 = FeatureDetector::create("HARRIS");	
	std::vector<cv::String> parameters;
	vidSIFTdetector1->getParams(parameters);
	for(int i2 = 0; i2 < parameters.size(); i2++)
	  {
	    cout <<  parameters[i2] << endl;
	  }
	  for(;;);*/
	  //End DEBUG
	



	/*SiftFeatureDetector sonSIFTdetector;//SIFTthreshold, SIFTedgeThreshold);
	SiftDescriptorExtractor sonSIFTextractor;
	FlannBasedMatcher sonSIFTmatcher;	
	SiftFeatureDetector vidSIFTdetector(SIFTthreshold, SIFTedgeThreshold);
OA	SiftDescriptorExtractor vidSIFTextractor;
	FlannBasedMatcher vidSIFTmatcher;
	*/

	//Create CLAHE:
	int CLAHEclipLimit = 4;//DEBUG should be 6
	cv::Size CLAHEtileGridSize(16,16);//was 16,16
	Ptr<CLAHE> clahe = createCLAHE(CLAHEclipLimit, CLAHEtileGridSize);
	//clahe->setClipLimit(clipLimit);


	//----------------------------------------------------------------------
	//--------------------Initialize Camera Detection ----------------------
	//----------------------------------------------------------------------

	//Check for camera angle:
	if(Windows_Data)
	  {
	    switch(sensorAngle)
	      {
	      case 0:
		camDataType = 4;
		break;
	      case 17:
		camDataType = 5;
		break;
	      case 30:
		camDataType = 6;
		break;
	      case 68:
		camDataType = 7;
		break;
	      default:
		cout << "WINDOWS CAM ANGLE NOT DETECTED. USING 30-DEG." << endl;
		camDataType = 6;
		break;
	      }
	  }
	else //Only one angle for linux data
	  camDataType = 2; //30 deg

	//Get the camera matrix and distortion coeffs for correct angle and camera
	Mat cameraMatrix;
	Mat distCoeffs;
	switch(camDataType)
	  {
	  case 0:
	    cout << "CAMERA ANGLE = 0-DEG LINUX" << endl;
	    cameraMatrix = Mat(3,3,DataType<double>::type,camArr0degLinux);
	    distCoeffs = Mat(5,1,DataType<double>::type,distArr0degLinux);
	    break;
	  case 1:
	    cout << "CAMERA ANGLE = 17-DEG LINUX" << endl;
	    cameraMatrix = Mat(3,3,DataType<double>::type,camArr17degLinux);
	    distCoeffs = Mat(5,1,DataType<double>::type,distArr17degLinux);
	    break;
	  case 2:
	    cout << "CAMERA ANGLE = 30-DEG LINUX" << endl;
	    cameraMatrix = Mat(3,3,DataType<double>::type,camArr30degLinux);
	    distCoeffs = Mat(5,1,DataType<double>::type,distArr30degLinux);
	    break;
	  case 3:
	    cout << "CAMERA ANGLE = 68-DEG LINUX" << endl;
	    cameraMatrix = Mat(3,3,DataType<double>::type,camArr68degLinux);
	    distCoeffs = Mat(5,1,DataType<double>::type,distArr68degLinux);
	    break;
	  case 4:
	    cout << "CAMERA ANGLE = 0-DEG WINDOWS" << endl;
	    cameraMatrix = Mat(3,3,DataType<double>::type,camArr0degWin);
	    distCoeffs = Mat(5,1,DataType<double>::type,distArr0degWin);
	    break;
	  case 5:
	    cout << "CAMERA ANGLE = 17-DEG WINDOWS" << endl;
	    cameraMatrix = Mat(3,3,DataType<double>::type,camArr17degWin);
	    distCoeffs = Mat(5,1,DataType<double>::type,distArr17degWin);
	    break;
	  case 6:
	    cout << "CAMERA ANGLE = 30-DEG WINDOWS" << endl;
	    cameraMatrix = Mat(3,3,DataType<double>::type,camArr30degWin);
	    distCoeffs = Mat(5,1,DataType<double>::type,distArr30degWin);
	    break;
	  case 7:
	    cout << "CAMERA ANGLE = 68-DEG LINUX" << endl;
	    cameraMatrix = Mat(3,3,DataType<double>::type,camArr68degWin);
	    distCoeffs = Mat(5,1,DataType<double>::type,distArr68degWin);
	    break;
	  default:
	    cout << "ERROR DETERMINING CAMERA + ANGLE. USING 30-DEG LINUX" << endl;
	    cameraMatrix = Mat(3,3,DataType<double>::type,camArr30degLinux);
	    distCoeffs = Mat(5,1,DataType<double>::type,distArr30degLinux);
	    break;
	  }

	//DEBUG: use laptop cam
	if(LAPTOP_CAMERA)
	  {
	    cameraMatrix = Mat(3,3,DataType<double>::type,camArrLaptop);
	    distCoeffs = Mat(5,1,DataType<double>::type,distArrLaptop);
	    cout << "DEBUG: USING LAPTOP CAM: " << endl << cameraMatrix << endl << distCoeffs << endl;
	    Windows_Data = false;
	    pings = numVidFrames;
	  }

	Mat cameraMatrix_transpose;
	transpose(cameraMatrix,cameraMatrix_transpose);

	//Create new camera matrix with 85 degree x FOV
	Mat newCameraMatrix = cameraMatrix.clone();
	double fx_new = (videoSize.width/2)/(tan((CV_PI/180)*((double)DES_FOVX)/2));
	double fx_old = cameraMatrix.at<double>(0,0);
	double fy_old = cameraMatrix.at<double>(1,1);
	double fy_new = (fx_new*fy_old)/fx_old;
	Point2d princ_pt = Point2d(cameraMatrix.at<double>(0,2),cameraMatrix.at<double>(1,2));
	newCameraMatrix.at<double>(0,0) = fx_new;
	newCameraMatrix.at<double>(1,1) = fy_new;

	cout << "New Cam Mat: " << newCameraMatrix << endl;
	cout << "fx ratio: " << fx_old/fx_new << endl;
	cout << "fy ratio: " << fy_old/fy_new << endl;
	cout << "princ_pt: " << princ_pt << endl;

	//Get undistortion rectify map:
	Mat map1, map2;
	//videoSize = Size(720,480); //DEBUG!!!
	cout << "VID SIZE: " << videoSize << endl;
	initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(), newCameraMatrix, videoSize, CV_32FC1, map1, map2);

	//-------------------------------------------------
	//Rotation matrices for rotating between camera and vehicle coordinates
	double camtoVehArr[3][3] = {0, 0, 1, 1, 0, 0, 0, 1, 0};
	double neg30deg = -30*CV_PI/180;
	double neg20deg = -20*CV_PI/180;
	double rotNeg30x[3][3] = {1, 0, 0, 0, cos(neg30deg), sin(neg30deg), 0, -sin(neg30deg), cos(neg30deg)};
	double rotNeg20x[3][3] = {1, 0, 0, 0, cos(neg20deg), sin(neg20deg), 0, -sin(neg20deg), cos(neg20deg)};
	Mat RcamtoV = Mat(3,3,DataType<double>::type,camtoVehArr);
	Mat Rneg30x = Mat(3,3,DataType<double>::type,rotNeg30x);
	Mat Rneg20x = Mat(3,3,DataType<double>::type,rotNeg20x);

	cout << "RcamtoV = " << endl << RcamtoV << endl;
	cout << "Rneg30x = " << endl << Rneg30x << endl;

	//---------------------------------------------------------------------
	//----------------Main Loop--------------------------------------------

	//Create Previous Image Containers:
	Mat prev_sonarImgGray, prev_videoImgGray, videoImgGray; 
	int prev_i = 0;

     for (i = 0; i < pings; i++) {
          cout << i << "/" << pings << endl;
	  // Now, get a ping!
	  BVTPing ping = NULL;
	  //if ( i == 0) {
	  if(LAPTOP_CAMERA)
	    ret = BVTHead_GetPing(head, 1, &ping);
	  else
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
	
	  // Generate an image from the ping:	  
	  if(SONAR_CARTESIAN)
	    ret = BVTPing_GetImageXY(ping, &bvtmag_img);
	  else //POLAR
	    ret = BVTPing_GetImageRTheta(ping, &bvtmag_img);
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
	  //cout << "---------------------" << endl << "SON_PIX_X = " << SON_PIX_X << endl;
	  Size sonarSize(width,height);
	  cout << "Sonar Size = " << sonarSize << endl;

	  // Create a IplImage header
	  color_img_ipl = cvCreateImageHeader(cvSize(width,height), IPL_DEPTH_8U, 4);
	
	  // And set it's data
	  cvSetImageData(color_img_ipl,  BVTColorImage_GetBits(bvtclr_img), width*4);

	//Convert IPL image to Mat:
	//cv::Mat colorImg(color_img_ipl);
	  Mat colorImg = cvarrToMat(color_img_ipl,true); //NEED TO GET RID OF OLD IPL IMG

	  //DEBUG:
	  //Mat  colorImg = Mat::zeros(height,width,CV_8UC4);
	    //END DEBUG

	//Get Ping Timestamp and find time from Ping0
	ping_time_sec = BVTPing_GetTimestamp(ping) - SONAR_TIME0;
	frame_num_windows = (ping_time_sec + SYNCH_OFFSET_SEC)*VIDEO_FPS;
	frame_num_windows_int = (int)round(frame_num_windows);

	if(frame_num_windows_int < 0) //Can't have negative frame num
	  frame_num_windows_int = 0;
	else if(frame_num_windows_int > (VIDEO_FRAME_COUNT-1))
	  frame_num_windows_int = (int)(VIDEO_FRAME_COUNT-1);

	//std::cout << i << " === " << ping_time_sec << " === " << frame_num_windows_int << std::endl;
	
	//-------------------------------------------

	inputVideo.set(CAP_PROP_POS_FRAMES,i);
	if(Windows_Data) //IF WINDOWS DATA, reset the position to calculated frame
	  {
	    inputVideo.set(CAP_PROP_POS_FRAMES, frame_num_windows_int);
	    //cout << "frame num win int = " << frame_num_windows_int << endl;	
	  }
	
	inputVideo >> videoImg;
	

	//std::cout << inputVideo.get(CAP_PROP_POS_MSEC) << std::endl;

	//FIX FOR BAD FRAMES AT END OF WINDOWS DATA
	while(videoImg.rows == 0)
	  {
	    frame_num_windows_int--;
	    inputVideo.set(CAP_PROP_POS_FRAMES, frame_num_windows_int);
	    inputVideo >> videoImg;
	  }
       

	//---------------------------------------------------------
	//-------------Preprocess Video Frame----------------------------
	//---------------------------------------------------------

	//Get gray image:
	cvtColor(videoImg, videoImgGray, COLOR_BGR2GRAY);

	imwrite("VideoIMG.jpg", videoImg);
	imwrite("VideoGrayIMG.jpg", videoImgGray);

	//Resize Image:
	//imshow(video_wnd, videoImgGray);
       	//resize(videoImgGray, videoImgGray, Size(720,480));
	//imshow(video_wnd, videoImgGray);

	//Apply CLAHE:
	clahe->apply(videoImgGray,videoImgGray); 
	imwrite("VideoCLAHEIMG.jpg", videoImgGray);

	//Undistort the image:
	Mat videoImgUndistort;
	remap(videoImgGray, videoImgUndistort, map1, map2, INTER_LINEAR);
	videoImgGray = videoImgUndistort.clone(); 
	imwrite("VideoUndistortIMG.jpg", videoImgGray);

	//Apply Blur
	Size vidBlurKernelSize = Size(7,7);
       	if(BLUR_VID_IMGS && (!LAPTOP_CAMERA))
	  GaussianBlur(videoImgGray,videoImgGray,vidBlurKernelSize,0,0); //Gaussian Blur
	imwrite("VideoImgBlur.jpg", videoImgGray);
	//------------------------------------------------------------
	//----------- Preprocess SONAR Frame --------------------------------------
	//------------------------------------------------------------

	//Get Gray Sonar Image:
	cvtColor(colorImg, sonarImgGray, COLOR_BGR2GRAY);

	//Remove Lines (cols 112-114 and 142-144) from Sonar Images:
	sonarImgGray.col(111).copyTo(sonarImgGray.col(112));
	line(sonarImgGray,Point(113,0),Point(113,sonarImgGray.rows-1),Scalar(0,0,0)); //Set the middle column to zero first ...
	
	sonarImgGray.col(113) += sonarImgGray.col(111)*0.5; // ... then add half of left col
	sonarImgGray.col(113) += sonarImgGray.col(115)*0.5; // ... then add half of left col to finally create an average of the two
	sonarImgGray.col(115).copyTo(sonarImgGray.col(114));

	sonarImgGray.col(141).copyTo(sonarImgGray.col(142));
	line(sonarImgGray,Point(143,0),Point(143,sonarImgGray.rows-1),Scalar(0,0,0)); //Set the middle column to zero first ...
        sonarImgGray.col(143) += sonarImgGray.col(141)*0.5; // ... then add half of left col                                                                            
	sonarImgGray.col(143) += sonarImgGray.col(145)*0.5; // ... then add half of left col to finally create an average of the two     
	sonarImgGray.col(145).copyTo(sonarImgGray.col(144));
       
	//Undistort Images:

	//Histogram Equalization with CLAHE:
	//clahe->apply(sonarImgGray,sonarImgGray);

	imwrite("SonarGrayIMG.jpg", sonarImgGray);

	//Apply Sonar Image Blurring:
	Size blurKernelSize = Size(5,5);
	if(BLUR_SON_IMGS)
	  GaussianBlur(sonarImgGray,sonarImgGray,blurKernelSize,0,0); //Gaussian BlurOB
	//blur(sonarImgGray,sonarImgGray,blurKernelSize,Point(-1,-1)); //Box Filter Blur
	/*
	//Debug create a known image:
	sonarImgGray = Scalar(0,0,0);
	rectangle(sonarImgGray,Point(100+i,100+i),Point(200+i,200+i),Scalar(255,255,255));*/

	imwrite("SonarBlurIMG.jpg", sonarImgGray);

	//Apply Thresholding:
	int son_thresh_level = 40;
	//int son_thresh_level = 160; //For CLAHE images
	int son_thresh_type = 3;
	if(THRESHOLD_SON_IMGS)
	  cv::threshold(sonarImgGray,sonarImgGray,son_thresh_level,max_BINARY_value,son_thresh_type);
	/*	
	//Higher threshold for middle and edge columns
	int higher_thresh_level = son_thresh_level + 0;
	Mat sonar_roi(sonarImgGray,Rect(112,0,30,sonarImgGray.rows));
	cv::threshold(sonar_roi,sonar_roi,higher_thresh_level,max_BINARY_value,son_thresh_type);
	Mat sonar_roi1(sonarImgGray,Rect(0,0,30,sonarImgGray.rows));
	cv::threshold(sonar_roi1,sonar_roi1,higher_thresh_level,max_BINARY_value,son_thresh_type);
	Mat sonar_roi2(sonarImgGray,Rect(225,0,30,sonarImgGray.rows));
	cv::threshold(sonar_roi2,sonar_roi2,higher_thresh_level,max_BINARY_value,son_thresh_type);
	*/

	//Fill in previous if i = 0:
	if(i == 0)
	  {
	    cout << "I is 0 - saving fake previous image" << endl;
	    prev_sonarImgGray = sonarImgGray.clone();
	    prev_videoImgGray = videoImgGray.clone();
	    prev_i = i;
	  }
	
	//-------------------------------------------------------------------
	//----------- Get SONAR Shifts --------------------------------------
	//-------------------------------------------------------------------

	//------------------ Sparse Optical Flow Estimates ---------------//
	// Parameters for Shi-Tomasi algorithm                                      
        vector<Point2f> sonOFcorners1;
	vector<Point2f> sonOFcorners2;
	vector<Point2f> sonOFpts_diff;
	vector<KeyPoint> sonOFkeypoints1;
	vector<KeyPoint> sonOFkeypoints2;
	Mat sonNewMatchesImg;
        double sonOFqualityLevel = 0.01;
	double sonOFminDistance = 10;
	int sonOFblockSize = 3;
        bool sonOFuseHarrisDetector = false;
        double sonOF_k = 0.04;
        int sonOF_r = 3;      //Radius of points for Corners 
	TermCriteria sonOFtermcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 20, 0.03);
        Size sonOFwinSize(31,31);
	int sonOFmax_corners = 100;
	Mat sonOFmodel, sonOFmask;
	float sonOFnumInliers = 0;
	//matchesImg = prev_sonarImgGray.clone();

	//Corner Detection:
	goodFeaturesToTrack( prev_sonarImgGray, sonOFcorners1, sonOFmax_corners, sonOFqualityLevel, sonOFminDistance, Mat(), sonOFblockSize, sonOFuseHarrisDetector, sonOF_k);

        if(sonOFcorners1.size() > 0)
	  {

	    //Calculate Corners Subpixel Accuracy:
	    Size sonOFsubPixWinSize(10,10);
	    cornerSubPix(prev_sonarImgGray, sonOFcorners1, sonOFsubPixWinSize, Size(-1,-1), sonOFtermcrit);

	    //Lucas Kanade Pyramid Algorithm:        
	    vector<uchar> sonOFstatus;
	    vector<float> sonOFerr;
	    //Mat sonOFpyrLKImg = sonarImgGray.clone();

	    calcOpticalFlowPyrLK(prev_sonarImgGray, sonarImgGray, sonOFcorners1, sonOFcorners2, sonOFstatus, sonOFerr, sonOFwinSize, 7, sonOFtermcrit, 0, 0.001);

	    //cout << "NUM OF CORNERS:" << sonOFcorners1.size() << endl;
	    std::vector< DMatch > sonOFgood_matches;
	    std::vector<Point2f> sonOFgood_match_pts1;
	    std::vector<Point2f> sonOFgood_match_pts2;

	    for(int i1=0; i1 < sonOFcorners1.size(); i1++)
	      {
		if(sonOFstatus[i1])
		  {
		    //cout << "corner1: " << sonOFcorners1[i1] << " -- corner2:" << sonOFcorners2[i1] << endl;
		    sonOFpts_diff.push_back(sonOFcorners2[i1] - sonOFcorners1[i1]);
		    int i_diff = sonOFpts_diff.size()-1;
		    //cout << i_diff << ": " << sonOFpts_diff[i_diff] << endl;
		    //Save Points into Keypoint Form For Drawing Matches:
		    sonOFkeypoints1.push_back(KeyPoint(sonOFcorners1[i1],1.f));
		    sonOFkeypoints2.push_back(KeyPoint(sonOFcorners2[i1],1.f));  //sonOFblockSize));
		    sonOFgood_match_pts1.push_back( sonOFcorners1[i1] );
		    sonOFgood_match_pts2.push_back( sonOFcorners2[i1] );
		    //float tmpDist = sqrt(sonOFpts_diff[i_diff].x*sonOFpts_diff[i_diff].x + sonOFpts_diff[i_diff].y*sonOFpts_diff[i_diff].y);
		    //sonOFgood_matches.push_back(DMatch(i_diff,i_diff,0,tmpDist));
		    //		    cout << sonOFcorners1[i1] << " --- " << sonOFcorners2[i1] << endl;
		  }
	      }
	if(sonOFpts_diff.size() >= 3) //If found some matches
	  {
	    //Run RANSAC                                                        
	    double sonOF_RANSAC_reprojthresh = 1;
	    double sonOF_RANSAC_param = 0.99;
	    int sonOFok;
	    if(SONAR_CARTESIAN)
	      sonOFok = estimateRigidTransform2DNew(sonOFgood_match_pts1,sonOFgood_match_pts2, ROBUST_EST_METH, sonOFmodel, sonOFmask, sonOF_RANSAC_reprojthresh, sonOF_RANSAC_param, sonarSize);
	    else //POLAR
	      sonOFok = estimateTranslationNew(sonOFpts_diff,sonOFpts_diff, ROBUST_EST_METH, sonOFmodel, sonOFmask, sonOF_RANSAC_reprojthresh, sonOF_RANSAC_param);

	    sonOFnumInliers = sum(sonOFmask)[0]; //Find number of inliers

            //Show Features:                                               
            cvtColor(prev_sonarImgGray, sonFeaturesImg, CV_GRAY2BGR); //Get copy of gray img to mark features      
	    //drawMatches(prev_sonarImgGray, sonOFkeypoints1, sonarImgGray, sonOFkeypoints2, sonOFgood_matches, matchesImg,Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            //cout << sonOFcorners1.size() << endl;

	    //Draw Feature Matches:
	    sonNewMatchesImg = Mat(sonarImgGray.rows,2*sonarImgGray.cols,CV_8U);
	    Point2f sonOFimg2offset = Point2f(sonarImgGray.cols,0);
	    Mat son_roi1(sonNewMatchesImg,Rect(0,0,sonarImgGray.cols,sonarImgGray.rows));
	    prev_sonarImgGray.copyTo(son_roi1);
      	    Mat son_roi2(sonNewMatchesImg,Rect(sonarImgGray.cols,0,sonarImgGray.cols,sonarImgGray.rows));
	    sonarImgGray.copyTo(son_roi2);
	    cvtColor(sonNewMatchesImg, sonMatchesImg, CV_GRAY2BGR);  

	    //Draw Inliers and matches:
            for(int i1 = 0; i1 < sonOFpts_diff.size(); i1++)
	      {
		Scalar color;
		if(sonOFmask.at<char>(0,i1))
		   color = Scalar(0,255,0);
		else
		  color = Scalar(0,0,255);
		circle(sonFeaturesImg, sonOFcorners1[i1], 4, color);
		//Draw Matches and lines between:
		circle(sonMatchesImg, sonOFkeypoints1[i1].pt, 4, color);
		circle(sonMatchesImg, sonOFkeypoints2[i1].pt + sonOFimg2offset, 4, color);
		line(sonMatchesImg, sonOFkeypoints1[i1].pt, sonOFkeypoints2[i1].pt+sonOFimg2offset, color);
		//	cout << sonOFcorners1[i1] << " --- " << sonOFcorners2[i1] << endl;
		//		cout << "corners2 new = " << sonOFcorners2[i1]+sonOFimg2offset << endl;
	      }

	  }
	else
	  sonOFmodel = Mat::zeros(1,2,CV_64F);
     
	  }
	else //If no corners found:
	  sonOFmodel = Mat::zeros(1,2,CV_64F);

	//cout << sonOFmask << endl;
	cout << "Num OFlow Matches: " << sonOFpts_diff.size() << endl;
	cout << "Num OFlow Inliers: " << sonOFnumInliers << endl;
	cout << "Sonar OFlow Model: " << sonOFmodel << endl;
	putText(sonFeaturesImg, "Optical Flow", Point(10,25),FONT_HERSHEY_SIMPLEX,1,Scalar(255,255,255));
	putText(sonMatchesImg, "Optical Flow", Point(10,25),FONT_HERSHEY_SIMPLEX,1,Scalar(255,255,255));
	if(DISPLAY_IMGS)
	  {
	    
	    imshow(son_features_wnd, sonFeaturesImg);
	    
	    imshow(son_matches_wnd, sonMatchesImg);
	    waitKey(0);
	    imwrite("SonarMatchesIMG.jpg", sonMatchesImg);
	    imwrite("SonarFeaturesIMG.jpg", sonFeaturesImg);
	  }
	//------------ SURF Feature Detection -----------------
	cvtColor(prev_sonarImgGray, sonFeaturesImg, CV_GRAY2BGR); //Get copy of gray img to mark features      
	sonMatchesImg = prev_sonarImgGray.clone();
	Mat sonSURFmodel, sonSURFmask;
	float sonSURFnumInliers = 0;
	float sonSURFnumMatches = 0;
	std::vector<KeyPoint> sonSURFkeypoints_1, sonSURFkeypoints_2;
	sonSURFdetector.detect( prev_sonarImgGray, sonSURFkeypoints_1 );
	sonSURFdetector.detect( sonarImgGray, sonSURFkeypoints_2 );
	//SURF Calculate descriptors (feature vectors):                             
	Mat sonSURFdescriptors_1, sonSURFdescriptors_2;
	sonSURFextractor.compute( prev_sonarImgGray, sonSURFkeypoints_1, sonSURFdescriptors_1 );
	sonSURFextractor.compute( sonarImgGray, sonSURFkeypoints_2, sonSURFdescriptors_2 );

	if((!sonSURFdescriptors_1.empty()) && (!sonSURFdescriptors_2.empty()))
	  {
	    //SURF Matching descriptor vectors using FLANN matcher 
	    std::vector< DMatch > sonSURFmatches;
	    sonSURFmatcher.match( sonSURFdescriptors_1, sonSURFdescriptors_2, sonSURFmatches );
	    double sonSURFmax_dist = 0; double sonSURFmin_dist = 100;
	    //-- Quick calculation of max and min distances between keypoints       
	    for( int i1 = 0; i1 < sonSURFdescriptors_1.rows; i1++ )
	      { double dist = sonSURFmatches[i1].distance;
		if( dist < sonSURFmin_dist ) sonSURFmin_dist = dist;
		if( dist > sonSURFmax_dist ) sonSURFmax_dist = dist;
	      }
	//-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
	//-- or a small arbitary value ( 0.02 ) in the event that min_dist is very  
	//-- small)   //-- PS.- radiusMatch can also be used here.                  
	    std::vector< DMatch > sonSURFgood_matches;
	    for( int i1 = 0; i1 < sonSURFdescriptors_1.rows; i1++ )
	      { if( sonSURFmatches[i1].distance <= max(2*sonSURFmin_dist, 0.02) )
		  { sonSURFgood_matches.push_back( sonSURFmatches[i1]); }
	      }   

	    std::vector<Point2f> sonSURFgood_match_pts1;
	    std::vector<Point2f> sonSURFgood_match_pts2;
	    std::vector<Point2f> sonSURFgood_match_pts_diff;
	    for( int i1 = 0; i1 < sonSURFgood_matches.size(); i1++ )
	      {
		//-- Get the keypoints from the good matches     
		sonSURFgood_match_pts1.push_back( sonSURFkeypoints_1[ sonSURFgood_matches[i1].queryIdx ].pt );
		sonSURFgood_match_pts2.push_back( sonSURFkeypoints_2[ sonSURFgood_matches[i1].trainIdx ].pt );
		sonSURFgood_match_pts_diff.push_back(sonSURFkeypoints_2[sonSURFgood_matches[i1].trainIdx].pt - sonSURFkeypoints_1[sonSURFgood_matches[i1].queryIdx].pt);
	      }
	    sonSURFnumMatches = sonSURFgood_match_pts1.size();
	    
	    for(int i1 = 0; i1 < sonSURFnumMatches; i1++)
	      {
		//cout << sonSURFgood_match_pts2[i1] << "-" << sonSURFgood_match_pts1[i1] << "=" << sonSURFgood_match_pts_diff[i1] << endl;
	      }
	    
	    //Run RANSAC 
	    double sonSURF_RANSAC_reprojthresh = 1;
	    double sonSURF_RANSAC_param = 0.99;
	    int sonSURFok;
	    if(sonSURFgood_matches.size() >= 3)
	      {
		if(SONAR_CARTESIAN)
		  sonSURFok = estimateRigidTransform2DNew(sonSURFgood_match_pts1, sonSURFgood_match_pts2, ROBUST_EST_METH,sonSURFmodel, sonSURFmask, sonSURF_RANSAC_reprojthresh, sonSURF_RANSAC_param, sonarSize);
		else //POLAR
		  sonSURFok = estimateTranslationNew(sonSURFgood_match_pts_diff, sonSURFgood_match_pts_diff, ROBUST_EST_METH, sonSURFmodel, sonSURFmask, sonSURF_RANSAC_reprojthresh, sonSURF_RANSAC_param);

		sonSURFnumInliers = sum(sonSURFmask)[0]; //Find number of inliers

		//Draw Features: 
		drawMatches(prev_sonarImgGray, sonSURFkeypoints_1, sonarImgGray, sonSURFkeypoints_2, sonSURFgood_matches, sonMatchesImg,Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		for(int i1 = 0; i1 < sonSURFgood_match_pts2.size(); i1++)
		  {
		    Scalar color;
		    if(sonSURFmask.at<char>(0,i1))
		      color = Scalar(0,255,0);
		    else
		      color = Scalar(0,0,255);
		    circle(sonFeaturesImg, sonSURFgood_match_pts1[i1], 4, color);
		  }
	      }

	    else
	      sonSURFmodel = Mat::zeros(1,2,CV_64F);

	  }
	else
	  sonSURFmodel = Mat::zeros(1,2,CV_64F);

	//cout << "Num SURF Matches: " << sonSURFnumMatches << endl;
	//cout << "Num SURF Inliers: " << sonSURFnumInliers << endl;
	cout << "Sonar SURF Model" << sonSURFmodel << endl;
	putText(sonFeaturesImg, "SURF", Point(10,25),FONT_HERSHEY_SIMPLEX,1,Scalar(255,255,255));
	putText(sonMatchesImg, "SURF", Point(10,25),FONT_HERSHEY_SIMPLEX,1,Scalar(255,255,255));
	if(DISPLAY_IMGS)
	  {
	    imshow(son_features_wnd, sonFeaturesImg);
	    imshow(son_matches_wnd, sonMatchesImg);
	    waitKey(0);
	  }

	//---------------Sonar SIFT Feature Detection ------------------//
	cvtColor(prev_sonarImgGray, sonFeaturesImg, CV_GRAY2BGR); //Get copy of gray img to mark features      
	sonMatchesImg = prev_sonarImgGray.clone();
        Mat sonSIFTmodel, sonSIFTmask;
	float sonSIFTnumInliers = 0;
	float sonSIFTnumMatches = 0;
	std::vector<KeyPoint> sonSIFTkeypoints_1, sonSIFTkeypoints_2;
        sonSIFTdetector->detect( prev_sonarImgGray, sonSIFTkeypoints_1 );
        sonSIFTdetector->detect( sonarImgGray, sonSIFTkeypoints_2 );
        //SURF Calculate descriptors (feature vectors):                             
        Mat sonSIFTdescriptors_1, sonSIFTdescriptors_2;
        sonSIFTextractor->compute( prev_sonarImgGray, sonSIFTkeypoints_1, sonSIFTdescriptors_1 );
        sonSIFTextractor->compute( sonarImgGray, sonSIFTkeypoints_2, sonSIFTdescriptors_2 );

	if((!sonSIFTdescriptors_1.empty()) && (!sonSIFTdescriptors_2.empty()))
	  {
	    //SURF Matching descriptor vectors using FLANN matcher                  
	    std::vector< DMatch > sonSIFTmatches;
	    sonSIFTmatcher.match( sonSIFTdescriptors_1, sonSIFTdescriptors_2, sonSIFTmatches );
	    double sonSIFTmax_dist = 0; double sonSIFTmin_dist = 100;
	    //-- Quick calculation of max and min distances between keypoints            
	    for( int i1 = 0; i1 < sonSIFTdescriptors_1.rows; i1++ )
	      { double dist = sonSIFTmatches[i1].distance;
		if( dist < sonSIFTmin_dist ) sonSIFTmin_dist = dist;
		if( dist > sonSIFTmax_dist ) sonSIFTmax_dist = dist;
	      }
	    //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,  or a small arbitary value ( 0.02 ) in the event that min_dist is very small)   //-- PS.- radiusMatch can also be used here. 
	    std::vector< DMatch > sonSIFTgood_matches;
	    for( int i1 = 0; i1 < sonSIFTdescriptors_1.rows; i1++ )
	      { if( sonSIFTmatches[i1].distance <= max(2*sonSIFTmin_dist, 0.02) )
		  { sonSIFTgood_matches.push_back( sonSIFTmatches[i1]); }
	      }
	    std::vector<Point2f> sonSIFTgood_match_pts1;
	    std::vector<Point2f> sonSIFTgood_match_pts2;
	    std::vector<Point2f> sonSIFTgood_match_pts_diff;


	    for( int i1 = 0; i1 < sonSIFTgood_matches.size(); i1++ )
	      {
		//-- Get the keypoints from the good matches                             
		sonSIFTgood_match_pts1.push_back( sonSIFTkeypoints_1[ sonSIFTgood_matches[i1].queryIdx ].pt );
		sonSIFTgood_match_pts2.push_back( sonSIFTkeypoints_2[ sonSIFTgood_matches[i1].trainIdx ].pt );
		sonSIFTgood_match_pts_diff.push_back( sonSIFTkeypoints_2[ sonSIFTgood_matches[i1].trainIdx].pt - sonSIFTkeypoints_1[ sonSIFTgood_matches[i1].queryIdx].pt);
	      }	  
	    sonSIFTnumMatches = sonSIFTgood_match_pts1.size();

	    for(int i1 = 0; i1 < sonSIFTgood_match_pts1.size(); i1++)
	      {
		//cout << sonSIFTgood_match_pts2[i1] << "-" << sonSIFTgood_match_pts1[i1] << "=" << sonSIFTgood_match_pts_diff[i1] << endl;
	      }

	    //Run RANSAC                                            
	    double sonSIFT_RANSAC_reprojthresh = 1;
	    double sonSIFT_RANSAC_param = 0.99;
	    int sonSIFTok;
	    if(sonSIFTgood_matches.size() >= 3)
	      {
		if(SONAR_CARTESIAN)
		  sonSIFTok = estimateRigidTransform2DNew(sonSIFTgood_match_pts1, sonSIFTgood_match_pts2, ROBUST_EST_METH, sonSIFTmodel, sonSIFTmask, sonSIFT_RANSAC_reprojthresh, sonSIFT_RANSAC_param, sonarSize);
		else //POLAR
		  sonSIFTok = estimateTranslationNew(sonSIFTgood_match_pts_diff, sonSIFTgood_match_pts_diff, ROBUST_EST_METH, sonSIFTmodel, sonSIFTmask, sonSIFT_RANSAC_reprojthresh, sonSIFT_RANSAC_param);

		sonSIFTnumInliers = sum(sonSIFTmask)[0]; //Find number of inliers
		
		//Draw Features:                
		drawMatches(prev_sonarImgGray, sonSIFTkeypoints_1, sonarImgGray, sonSIFTkeypoints_2, sonSIFTgood_matches, sonMatchesImg,Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);                      
		for(int i1 = 0; i1 < sonSIFTgood_match_pts1.size(); i1++)
		  {
		    Scalar color;
		    if(sonSIFTmask.at<char>(0,i1))
		      color = Scalar(0,255,0);
		    else
		      color = Scalar(0,0,255);
		    circle(sonFeaturesImg, sonSIFTgood_match_pts1[i1], 4, color);
		  }
		
	      }
	    else
	      sonSIFTmodel = Mat::zeros(1,2,CV_64F);
	    
	  }
	else 
	  sonSIFTmodel = Mat::zeros(1,2,CV_64F);

	//cout << "Num SIFT Matches: " << sonSIFTnumMatches << endl;
	//cout << "Num SIFT Inliers: " << sonSIFTnumInliers << endl;
	cout << "sonar SIFT model" << sonSIFTmodel << endl;
	putText(sonFeaturesImg, "SIFT", Point(10,25),FONT_HERSHEY_SIMPLEX,1,Scalar(255,255,255));
	putText(sonMatchesImg, "SIFT", Point(10,25),FONT_HERSHEY_SIMPLEX,1,Scalar(255,255,255));
	if(DISPLAY_IMGS)
	  {
	    imshow(son_features_wnd, sonFeaturesImg);    
	    imshow(son_matches_wnd, sonMatchesImg);
	    waitKey(0);
	  }

	//-------------------------- HARRIS/SIFT Detection ---------------//
	//-- create detector and descriptor --
	// if you want it faster, take e.g. FAST or ORB
	Mat sonHARRISmodel, sonHARRISmask;
	float sonHARRISnumMatches = 0;
	float sonHARRISnumInliers = 0;

	cvtColor(prev_sonarImgGray, sonFeaturesImg, CV_GRAY2BGR); //Get copy of gray img to mark features
	sonMatchesImg = prev_sonarImgGray.clone();

	cv::Ptr<cv::FeatureDetector> sonHARRISdetector = cv::FeatureDetector::create("HARRIS"); 
	// if you want it faster take e.g. ORB or FREAK
	cv::Ptr<cv::DescriptorExtractor> sonHARRISdescriptor = cv::DescriptorExtractor::create("SIFT"); 

	// detect keypoints
	std::vector<cv::KeyPoint> sonHARRISkeypoints1, sonHARRISkeypoints2;
	sonHARRISdetector->detect(prev_sonarImgGray, sonHARRISkeypoints1);
	sonHARRISdetector->detect(sonarImgGray, sonHARRISkeypoints2);

	// extract features
	cv::Mat sonHARRISdesc1, sonHARRISdesc2;
	sonHARRISdescriptor->compute(prev_sonarImgGray, sonHARRISkeypoints1, sonHARRISdesc1);
	sonHARRISdescriptor->compute(sonarImgGray, sonHARRISkeypoints2, sonHARRISdesc2);

	if((!sonHARRISdesc1.empty()) && (!sonHARRISdesc2.empty()))
	  {
	    //SURF Matching descriptor vectors using FLANN matcher                      
	    FlannBasedMatcher sonHARRISmatcher;
	    std::vector< DMatch > sonHARRISmatches;
	    sonHARRISmatcher.match( sonHARRISdesc1, sonHARRISdesc2, sonHARRISmatches );
	    double sonHARRISmax_dist = 0; double sonHARRISmin_dist = 100;
	    //-- Quick calculation of max and min distances between keypoints
	    for( int i1 = 0; i1 < sonHARRISdesc1.rows; i1++ )
	      { double dist = sonHARRISmatches[i1].distance;
		if( dist < sonHARRISmin_dist ) sonHARRISmin_dist = dist;
		if( dist > sonHARRISmax_dist ) sonHARRISmax_dist = dist;
	      }
	    //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist, 
	    //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very  
	    //-- small)   //-- PS.- radiusMatch can also be used here.                  
	    std::vector< DMatch > sonHARRISgood_matches;
	    for( int i1 = 0; i1 < sonHARRISdesc1.rows; i1++ )
	      { if( sonHARRISmatches[i1].distance <= max(2*sonHARRISmin_dist, 0.02) )
		  { sonHARRISgood_matches.push_back( sonHARRISmatches[i1]); }
	      }
	    std::vector<Point2f> sonHARRISgood_match_pts1;
	    std::vector<Point2f> sonHARRISgood_match_pts2;
	    std::vector<Point2f> sonHARRISgood_match_pts_diff;
	    
	    
	    for( int i1 = 0; i1 < sonHARRISgood_matches.size(); i1++ )
	      {
		//-- Get the keypoints from the good matches                            
		sonHARRISgood_match_pts1.push_back( sonHARRISkeypoints1[ sonHARRISgood_matches[i1].queryIdx ].pt );
		sonHARRISgood_match_pts2.push_back( sonHARRISkeypoints2[ sonHARRISgood_matches[i1].trainIdx ].pt );
		sonHARRISgood_match_pts_diff.push_back(sonHARRISkeypoints2[sonHARRISgood_matches[i1].trainIdx].pt - sonHARRISkeypoints1[sonHARRISgood_matches[i1].queryIdx].pt);
	      }
	    sonHARRISnumMatches = sonHARRISgood_match_pts1.size();

	    for(int i1 = 0; i1 < sonHARRISgood_match_pts1.size(); i1++)
	      {
		//cout << sonHARRISgood_match_pts2[i1] << "-" << sonHARRISgood_match_pts1[i1] << "=" << sonHARRISgood_match_pts_diff[i1] << endl;
	      }

	    	    
      	    //Run RANSAC                                                            
	    double sonHARRIS_RANSAC_reprojthresh = 1;
	    double sonHARRIS_RANSAC_param = 0.99;
	    int sonHARRISok;
	    if(sonHARRISgood_matches.size() > 3)
	      {
		if(SONAR_CARTESIAN)
		  sonHARRISok = estimateRigidTransform2DNew(sonHARRISgood_match_pts1, sonHARRISgood_match_pts2, ROBUST_EST_METH, sonHARRISmodel, sonHARRISmask, sonHARRIS_RANSAC_reprojthresh, sonHARRIS_RANSAC_param, sonarSize);
		else //POLAR
		  sonHARRISok = estimateTranslationNew(sonHARRISgood_match_pts_diff, sonHARRISgood_match_pts_diff, ROBUST_EST_METH, sonHARRISmodel, sonHARRISmask, sonHARRIS_RANSAC_reprojthresh, sonHARRIS_RANSAC_param);

		sonHARRISnumInliers = sum(sonHARRISmask)[0]; //Find number of inliers
		//Draw Features:
		drawMatches(prev_sonarImgGray, sonHARRISkeypoints1, sonarImgGray, sonHARRISkeypoints2, sonHARRISgood_matches, sonMatchesImg,Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		for(int i1 = 0; i1 < sonHARRISgood_match_pts1.size(); i1++)
		  {
		    Scalar color;
		    if(sonHARRISmask.at<char>(0,i1))
		      color = Scalar(0,255,0);
		    else
		      color = Scalar(0,0,255);
		    circle(sonFeaturesImg, sonHARRISgood_match_pts1[i1], 4, color);
		  }
	      }

	    else
	      sonHARRISmodel = Mat::zeros(1,2,CV_64F);
	    

	  }
	else sonHARRISmodel = Mat::zeros(1,2,CV_64F);


	//cout << "Num HARRIS Matches: " << sonHARRISnumMatches << endl;
	//cout << "Num HARRIS Inliers: " << sonHARRISnumInliers << endl;
	cout << "Sonar HARRIS Model: " << sonHARRISmodel << endl;
	putText(sonFeaturesImg, "HARRIS", Point(10,25),FONT_HERSHEY_SIMPLEX,1,Scalar(255,255,255));
	putText(sonMatchesImg, "HARRIS", Point(10,25),FONT_HERSHEY_SIMPLEX,1,Scalar(255,255,255));

	//SONAR GLIB ERROR HERE::: But other places too. Seems to be a problem with the python?? libraries or where opencv is installed. check it out by googling exact error message.
	if(DISPLAY_IMGS)
	  {
	    imshow(son_features_wnd, sonFeaturesImg);
	    imshow(son_matches_wnd, sonMatchesImg);
	    waitKey(0);
	  }
	//END SONAR GLIB ERROR

	//----------------------------------------------------------
	//-----------------Get Video Shifts-------------------------
	//----------------------------------------------------------

	//------------------ Sparse Optical Flow Estimates ---------------//
	// Parameters for Shi-Tomasi algorithm                                      
        vector<Point2f> vidOFcorners1;
	vector<Point2f> vidOFcorners2;
	//vector<Point2f> vidOFpts_diff;
	vector<Point2f> vidOFgoodMatches1;
	vector<Point2f> vidOFgoodMatches2;
	vector<KeyPoint> vidOFkeypoints1;
	vector<KeyPoint> vidOFkeypoints2;
	Mat vidNewMatchesImg;
        double vidOFqualityLevel = 0.01;
	double vidOFminDistance = 10;
	int vidOFblockSize = 3;
        bool vidOFuseHarrisDetector = false;
        double vidOF_k = 0.04;
        int vidOF_r = 3;      //Radius of points for Corners 
	TermCriteria vidOFtermcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 20, 0.03);
        Size vidOFwinSize(31,31);
	int vidOFmax_corners = 100;
	Mat vidOFmodel, vidOFmask;
	float vidOFnumInliers = 0;
	double vidOFtheta_x, vidOFtheta_y, vidOFtheta_z;
	Mat vidOFrotMat;
	Mat vidOFtranslMat;
	double vidOFanglesArr[3] = {0,0,0};
	Mat rotatedVidOFangles = Mat::zeros(3,1,DataType<double>::type);
	//vidMatchesImg = prev_videoImgGray.clone();

	//Corner Detection:
	goodFeaturesToTrack( prev_videoImgGray, vidOFcorners1, vidOFmax_corners, vidOFqualityLevel, vidOFminDistance, Mat(), vidOFblockSize, vidOFuseHarrisDetector, vidOF_k);

        if(vidOFcorners1.size() > 0)
	  {

	    //Calculate Corners Subpixel Accuracy:
	    Size vidOFsubPixWinSize(10,10);
	    cornerSubPix(prev_videoImgGray, vidOFcorners1, vidOFsubPixWinSize, Size(-1,-1), vidOFtermcrit);

	    //Lucas Kanade Pyramid Algorithm:        
	    vector<uchar> vidOFstatus;
	    vector<float> vidOFerr;
	    //Mat vidOFpyrLKImg = videoImgGray.clone();

	    calcOpticalFlowPyrLK(prev_videoImgGray, videoImgGray, vidOFcorners1, vidOFcorners2, vidOFstatus, vidOFerr, vidOFwinSize, 7, vidOFtermcrit, 0, 0.001);

	    //cout << "NUM VID OFlow CORNERS:" << vidOFcorners1.size() << endl;
	    std::vector< DMatch > vidOFgood_matches;
	    for(int i1=0; i1 < vidOFcorners1.size(); i1++)
	      {
		if(vidOFstatus[i1])
		  {
		    //cout << "corner1: " << vidOFcorners1[i1] << " -- corner2:" << vidOFcorners2[i1] << endl;
		    //vidOFpts_diff.push_back(vidOFcorners2[i1] - vidOFcorners1[i1]);
		    //int i_diff = vidOFpts_diff.size()-1;
		    //cout << i_diff << ": " << vidOFpts_diff[i_diff] << endl;
		    //Save Points into Keypoint Form For Drawing Matches:
		    vidOFkeypoints1.push_back(KeyPoint(vidOFcorners1[i1],1.f));
		    vidOFkeypoints2.push_back(KeyPoint(vidOFcorners2[i1],1.f));  //sonOFblockSize));
		    vidOFgoodMatches1.push_back(vidOFcorners1[i1]);
		    vidOFgoodMatches2.push_back(vidOFcorners2[i1]);
		    //float tmpDist = sqrt(vidOFpts_diff[i_diff].x*vidOFpts_diff[i_diff].x + vidOFpts_diff[i_diff].y*vidOFpts_diff[i_diff].y);
		    //vidOFgood_matches.push_back(DMatch(i_diff,i_diff,0,tmpDist));
		    //    cout << vidOFcorners1[i1] << " --- " << vidOFcorners2[i1] << endl;
		  }
	      }
	    if(vidOFgoodMatches1.size() > 5) //If found some matches
	      {
		//Find Fundamental and Essential Matrices
		double vidOF_RANSAC_reprojthresh = 1;
		double vidOF_RANSAC_param = 0.99;

		//Mat vidOFfundamentalMatrix = findFundamentalMat(vidOFgoodMatches1, vidOFgoodMatches2, ROBUST_EST_METH, vidOF_RANSAC_reprojthresh, vidOF_RANSAC_param,vidOFmask);
		//Mat temp_mask1;
		Mat vidOFessentialMatrix = findEssentialMatNew(vidOFgoodMatches1, vidOFgoodMatches2, fx_new, fy_new, princ_pt, ROBUST_EST_METH, vidOF_RANSAC_param, vidOF_RANSAC_reprojthresh,vidOFmask); 
		//Mat vidOFessentialMatrix = findEssentialMat(vidOFgoodMatches1, vidOFgoodMatches2, fx_new, princ_pt, CV_RANSAC, vidOF_RANSAC_param, vidOF_RANSAC_reprojthresh,vidOFmask);
		if(DISPLAY_POSE)
		  cout << "Essential Mat 1 = " << vidOFessentialMatrix << endl;
		Mat tempMask = vidOFmask.clone();

		if(vidOFessentialMatrix.rows > 0)
		  {
		    if(vidOFessentialMatrix.rows > 3)
		      {
			cout << "OF: " << vidOFessentialMatrix.rows/3 << " Essential Matrices Found" << endl;
			vidOFessentialMatrix.resize(3);
		      }
		    vidOFmodel = vidOFessentialMatrix.clone();
		    vidOFnumInliers = sum(vidOFmask)[0]; //Find number of inliers
		    if(DISPLAY_POSE)
		      cout << "vid OF num inliers = " << vidOFnumInliers << endl;
		  }
		else 
		  {
		    cout << "No Vid OF Model Found" << endl;
		    vidOFessentialMatrix = Mat::zeros(3,3,CV_64F);
		    vidOFmodel = Mat::zeros(3,3,CV_64F);
		    vidOFmask = Mat::zeros(1,vidOFgoodMatches1.size(),CV_8U);
		    vidOFnumInliers = 0;
		  }
		
		/******DEBUG:**/
		Mat tmpR1,tmpR2,tmpT;
		decomposeEssentialMat(vidOFessentialMatrix,tmpR1, tmpR2, tmpT);
		//cout << "OF TMP R1 = " << endl << tmpR1 << endl;
		//cout << "OF TMP R2 = " << endl << tmpR2 << endl;
		//cout << "OF TMP T = " << endl << tmpT << endl;
		//END DEBUG
 
		recoverPoseNew(vidOFessentialMatrix, vidOFgoodMatches1, vidOFgoodMatches2, vidOFrotMat, vidOFtranslMat, fx_new, fy_new, princ_pt, tempMask);
		
		//DEBUG
		Mat vidOFrotMat = tmpR1;
		Mat vidOFrotMatNew = tmpR2;
		//END OF DEBUG

		vidOFtheta_x = atan2(vidOFrotMat.at<double>(2,1),vidOFrotMat.at<double>(2,2))*180/CV_PI;
	  	vidOFtheta_y = atan2(-vidOFrotMat.at<double>(2,0),sqrt(vidOFrotMat.at<double>(2,1)*vidOFrotMat.at<double>(2,1) + vidOFrotMat.at<double>(2,2)*vidOFrotMat.at<double>(2,2)))*180/CV_PI;
		vidOFtheta_z = atan2(vidOFrotMat.at<double>(1,0),vidOFrotMat.at<double>(0,0))*180/CV_PI;  

		double vidOFtheta_x2 = atan2(vidOFrotMatNew.at<double>(2,1),vidOFrotMatNew.at<double>(2,2))*180/CV_PI;
	  	double vidOFtheta_y2 = atan2(-vidOFrotMatNew.at<double>(2,0),sqrt(vidOFrotMatNew.at<double>(2,1)*vidOFrotMatNew.at<double>(2,1) + vidOFrotMatNew.at<double>(2,2)*vidOFrotMatNew.at<double>(2,2)))*180/CV_PI;
		double vidOFtheta_z2 = atan2(vidOFrotMatNew.at<double>(1,0),vidOFrotMatNew.at<double>(0,0))*180/CV_PI;  
		
		//Check to find the correct rotation - if magnitude of x and z rot is bigger, then it is the wrong matrix. Also check if any angles larger than 100 (bigger than FOV) which means wrong matrix. THIS IS NOT PROVEN
		int VALID_ROT_MAT_FLAG = 0;
		if((abs(vidOFtheta_x) > abs(vidOFtheta_x2))&&(abs(vidOFtheta_z) > abs(vidOFtheta_z2)))
		  //cout << "****USE ANGLES 2" << endl;
		  VALID_ROT_MAT_FLAG = 2;
		else
		  VALID_ROT_MAT_FLAG = 1;
		if((abs(vidOFtheta_x) > 100) || (abs(vidOFtheta_y) > 100) || (abs(vidOFtheta_z) > 100))
		  //cout << "****USE ANGLES 2 AGAIN" << endl;
		  VALID_ROT_MAT_FLAG += 20;
		else if((abs(vidOFtheta_x2) > 100) || (abs(vidOFtheta_y2) > 100) || (abs(vidOFtheta_z2) > 100))
		  //cout << "****USE ANGLES 1" << endl;
		  VALID_ROT_MAT_FLAG += 10;

		//Choose between the two possible rotation matrices based on flag
		
		if((VALID_ROT_MAT_FLAG == 1) || (VALID_ROT_MAT_FLAG == 11))
		  { 
		    vidOFanglesArr[0] = vidOFtheta_x;
		    vidOFanglesArr[1] = vidOFtheta_y;
		    vidOFanglesArr[2] = vidOFtheta_z;
		  }
		else if((VALID_ROT_MAT_FLAG == 2) || (VALID_ROT_MAT_FLAG == 22))
		  { 
		    vidOFanglesArr[0] = vidOFtheta_x2;
		    vidOFanglesArr[1] = vidOFtheta_y2;
		    vidOFanglesArr[2] = vidOFtheta_z2;
		  }		   

		Mat vidOFangles = Mat(3,1,DataType<double>::type,vidOFanglesArr);
		rotatedVidOFangles = RcamtoV*Rneg30x*vidOFangles;

		if(LAPTOP_CAMERA)
		  rotatedVidOFangles = RcamtoV*Rneg20x*vidOFangles;

		//Get the sum angles
		vidOFsumAngles = vidOFsumAngles + rotatedVidOFangles;

		//cout << "vidOFRot = " << vidOFrotMat << endl;
		if(DISPLAY_POSE)
		  {
		    cout << "OF VALID ROT MAT FLAG = " << endl;
		    cout << "vidOFangles = " << vidOFangles << endl;
		    cout << "rotatedVidOFangles = " << rotatedVidOFangles << endl;
		    cout << "SumrotatedVidOFangles = " << vidOFsumAngles << endl;
		  }
		//cout << "Rotation: " << vidOFrotMat << endl;
		//cout << "Video OF Translation: " << vidOFtranslMat << endl;
		/*cout << "Video OF Thetax: " << vidOFtheta_x << " -- " << vidOFtheta_x2 << endl;
		cout << "Video OF Thetay: " << vidOFtheta_y <<  " -- " << vidOFtheta_y2 << endl;
		cout << "Video OF Thetaz: " << vidOFtheta_z <<  " -- " << vidOFtheta_z2 << endl;*/

		/*
		if(vidOFfundamentalMatrix.rows > 0)
		  {
		    if(vidOFfundamentalMatrix.rows > 3) //Should make better soln to multiple solutions returned
		      {
			vidOFfundamentalMatrix.resize(3);
			//cout << vidOFfundamentalMatrix << endl;
		      }
		    //cout << "OF fundmatrows = " << vidOFfundamentalMatrix.rows << endl;
		    Mat vidOFessentialMatrix = cameraMatrix_transpose*vidOFfundamentalMatrix*cameraMatrix;
		    vidOFmodel = vidOFessentialMatrix.clone();
		    vidOFnumInliers = sum(vidOFmask)[0]; //Find number of inliers
		  }
		else
		  {
		    vidOFmodel = Mat::zeros(3,3,CV_64F);
		    vidOFmask = Mat::zeros(1,vidOFgoodMatches1.size(),CV_8U);
		    vidOFnumInliers = 0;
		    //cout << "NO OF FUND MATRIX FOUND" << endl;
		  }
		*/

		//Show Features:                                               
		cvtColor(prev_videoImgGray, vidFeaturesImg, CV_GRAY2BGR); //Get copy of gray img to mark features      
		//drawMatches(prev_sonarImgGray, sonOFkeypoints1, sonarImgGray, sonOFkeypoints2, sonOFgood_matches, matchesImg,Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		//cout << vidOFcorners1.size() << endl;

		//Draw Feature Matches:
		vidNewMatchesImg = Mat(videoImgGray.rows,2*videoImgGray.cols,CV_8U);
		Point2f vidOFimg2offset = Point2f(videoImgGray.cols,0);
		Mat vid_roi1(vidNewMatchesImg,Rect(0,0,videoImgGray.cols,videoImgGray.rows));
		prev_videoImgGray.copyTo(vid_roi1);
		Mat vid_roi2(vidNewMatchesImg,Rect(videoImgGray.cols,0,videoImgGray.cols,videoImgGray.rows));
		videoImgGray.copyTo(vid_roi2);
		cvtColor(vidNewMatchesImg, vidMatchesImg, CV_GRAY2BGR); //To Color

		//Draw Inliers and matches:
		for(int i1 = 0; i1 < vidOFgoodMatches1.size(); i1++)
		  {
		    Scalar color;
		    //cout << (int)vidOFmask.type() << "===" << CV_8U << endl;
		    if(vidOFmask.at<char>(0,i1))
		      color = Scalar(0,255,0);
		    else
		      color = Scalar(0,0,255);
		    circle(vidFeaturesImg, vidOFcorners1[i1], 4, color);
		    //Draw Matches and lines between:
		    circle(vidMatchesImg, vidOFgoodMatches1[i1], 4, color);
		    circle(vidMatchesImg, vidOFgoodMatches2[i1] + vidOFimg2offset, 4, color);
		    line(vidMatchesImg, vidOFgoodMatches1[i1], vidOFgoodMatches2[i1]+vidOFimg2offset, color);
		    //cout << vidOFcorners1[i1] << " --- " << vidOFcorners2[i1] << endl;
		    //cout << "corners2 new = " << vidOFcorners2[i1]+vidOFimg2offset << endl;
		  }

	      }
	    else
	      vidOFmodel = Mat::zeros(3,3,CV_64F);
     
	  }
	else //If no corners found:
	  vidOFmodel = Mat::zeros(3,3,CV_64F);

	//cout << vidOFmask << endl;
	//cout << "Num Video OFlow Matches: " << vidOFgoodMatches1.size() << endl;
	//cout << "Num Video OFlow Inliers: " << vidOFnumInliers << endl;
	//cout << "Video OFlow Model: " << vidOFmodel << endl;
	putText(vidFeaturesImg, "Optical Flow (Video)", Point(10,25),FONT_HERSHEY_SIMPLEX,1,Scalar(255,255,255));
	putText(vidMatchesImg, "Optical Flow (Video)", Point(10,25),FONT_HERSHEY_SIMPLEX,1,Scalar(255,255,255));
	if(DISPLAY_IMGS)
	  {
	    //imshow(vid_features_wnd, vidFeaturesImg);
	    //imshow(vid_matches_wnd, vidMatchesImg);
	    waitKey(0);
	  }
	
	//------------ SURF Feature Detection -----------------
	cvtColor(prev_videoImgGray, vidFeaturesImg, CV_GRAY2BGR); //Get copy of gray img to mark features      
	vidMatchesImg = prev_videoImgGray.clone();
	Mat vidSURFmodel, vidSURFmask;
	float vidSURFnumInliers = 0;
	float vidSURFnumMatches = 0;
	double vidSURFtheta_x, vidSURFtheta_y, vidSURFtheta_z;
	Mat vidSURFrotMat;
	Mat vidSURFtranslMat;
	double vidSURFanglesArr[3] = {0,0,0};
	Mat rotatedVidSURFangles = Mat::zeros(3,1,DataType<double>::type);

	std::vector<KeyPoint> vidSURFkeypoints_1, vidSURFkeypoints_2;
	vidSURFdetector.detect( prev_videoImgGray, vidSURFkeypoints_1 );
	vidSURFdetector.detect( videoImgGray, vidSURFkeypoints_2 );
	//SURF Calculate descriptors (feature vectors):                             
	Mat vidSURFdescriptors_1, vidSURFdescriptors_2;
	vidSURFextractor.compute( prev_videoImgGray, vidSURFkeypoints_1, vidSURFdescriptors_1 );
	vidSURFextractor.compute( videoImgGray, vidSURFkeypoints_2, vidSURFdescriptors_2 );

	if((!vidSURFdescriptors_1.empty()) && (!vidSURFdescriptors_2.empty()))
	  {
	    //SURF Matching descriptor vectors using FLANN matcher 
	    std::vector< DMatch > vidSURFmatches;
	    vidSURFmatcher.match( vidSURFdescriptors_1, vidSURFdescriptors_2, vidSURFmatches );
	    double vidSURFmax_dist = 0; double vidSURFmin_dist = 100;
	    //-- Quick calculation of max and min distances between keypoints       
	    for( int i1 = 0; i1 < vidSURFdescriptors_1.rows; i1++ )
	      { double dist = vidSURFmatches[i1].distance;
		if( dist < vidSURFmin_dist ) vidSURFmin_dist = dist;
		if( dist > vidSURFmax_dist ) vidSURFmax_dist = dist;
	      }
	    //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
	    //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very  
	    //-- small)   //-- PS.- radiusMatch can also be used here.                  
	    std::vector< DMatch > vidSURFgood_matches;
	    for( int i1 = 0; i1 < vidSURFdescriptors_1.rows; i1++ )
	      { //if( vidSURFmatches[i1].distance <= max(2*vidSURFmin_dist, 0.02) )
		  { vidSURFgood_matches.push_back( vidSURFmatches[i1]); }
	      }   

	    std::vector<Point2f> vidSURFgood_match_pts1;
	    std::vector<Point2f> vidSURFgood_match_pts2;
	    std::vector<Point2f> vidSURFgood_match_pts_diff;
	    for( int i1 = 0; i1 < vidSURFgood_matches.size(); i1++ )
	      {
		//-- Get the keypoints from the good matches     
		vidSURFgood_match_pts1.push_back( vidSURFkeypoints_1[ vidSURFgood_matches[i1].queryIdx ].pt );
		vidSURFgood_match_pts2.push_back( vidSURFkeypoints_2[ vidSURFgood_matches[i1].trainIdx ].pt );
		vidSURFgood_match_pts_diff.push_back(vidSURFkeypoints_2[vidSURFgood_matches[i1].trainIdx].pt - vidSURFkeypoints_1[vidSURFgood_matches[i1].queryIdx].pt);
	      }
	    vidSURFnumMatches = vidSURFgood_match_pts1.size();
	        
	    for(int i1 = 0; i1 < vidSURFnumMatches; i1++)
	      {
		//cout << vidSURFgood_match_pts2[i1] << "-" << vidSURFgood_match_pts1[i1] << "=" << vidSURFgood_match_pts_diff[i1] << endl;
	      }
	        
	    //Find Fundamental and Essential Matrices
	    double vidSURF_RANSAC_reprojthresh = 1;
	    double vidSURF_RANSAC_param = 0.99;

	    if(vidSURFgood_matches.size() > 5)
	      {
		//vidSURFok = estimateTranslationNew(vidSURFgood_match_pts_diff, vidSURFgood_match_pts_diff, vidSURFmodel, vidSURFmask, vidSURF_RANSAC_reprojthresh, vidSURF_RANSAC_param);
		//Mat vidSURFfundamentalMatrix = findFundamentalMat(vidSURFgood_match_pts1, vidSURFgood_match_pts2, ROBUST_EST_METH, vidSURF_RANSAC_reprojthresh, vidSURF_RANSAC_param, vidSURFmask);

		//Mat temp_mask1;
		Mat vidSURFessentialMatrix = findEssentialMatNew(vidSURFgood_match_pts1, vidSURFgood_match_pts2, fx_new, fy_new, princ_pt, ROBUST_EST_METH, vidSURF_RANSAC_param, vidSURF_RANSAC_reprojthresh,vidSURFmask); //ADD MASK LATER
		//cout << "SURF Essential Mat = " << vidSURFessentialMatrix << endl;
		Mat tempMask = vidSURFmask.clone();

		if(vidSURFessentialMatrix.rows > 0)
		  {
		    if(vidSURFessentialMatrix.rows > 3)
		      {
			cout << "SURF: " << vidSURFessentialMatrix.rows/3 << " Essential Matrices Found" << endl;
			vidSURFessentialMatrix.resize(3);
		      }
		    vidSURFmodel = vidSURFessentialMatrix.clone();
		    vidSURFnumInliers = sum(vidSURFmask)[0]; //Find number of inliers
				
		  }
		else
		  {
		    vidSURFessentialMatrix = Mat::zeros(3,3,CV_64F);
		    vidSURFmodel = Mat::zeros(3,3,CV_64F);
		    vidSURFmask = Mat::zeros(1,vidSURFgood_match_pts2.size(),CV_8U);
		    vidSURFnumInliers = 0;
		    cout << "NO SURF FUNDAMENTAL MATRIX FOUND" << endl;
		  }
		
		Mat tmpR1,tmpR2,tmpT;
		decomposeEssentialMat(vidSURFessentialMatrix,tmpR1, tmpR2, tmpT);
		//cout << "SURF TMP R1 = " << endl << tmpR1 << endl;
		//cout << "SURF TMP R2 = " << endl << tmpR2 << endl;
		//cout << "SURF TMP T = " << endl << tmpT << endl;

		recoverPoseNew(vidSURFessentialMatrix, vidSURFgood_match_pts1, vidSURFgood_match_pts2, vidSURFrotMat, vidSURFtranslMat, fx_new, fy_new, princ_pt, tempMask);

		Mat vidSURFrotMat = tmpR1;
		Mat vidSURFrotMatNew = tmpR2;

		vidSURFtheta_x = atan2(vidSURFrotMat.at<double>(2,1),vidSURFrotMat.at<double>(2,2))*180/CV_PI;
		vidSURFtheta_y = atan2(-vidSURFrotMat.at<double>(2,0),sqrt(vidSURFrotMat.at<double>(2,1)*vidSURFrotMat.at<double>(2,1) + vidSURFrotMat.at<double>(2,2)*vidSURFrotMat.at<double>(2,2)))*180/CV_PI;
		vidSURFtheta_z = atan2(vidSURFrotMat.at<double>(1,0),vidSURFrotMat.at<double>(0,0))*180/CV_PI;  
		
		double vidSURFtheta_x2 = atan2(vidSURFrotMatNew.at<double>(2,1),vidSURFrotMatNew.at<double>(2,2))*180/CV_PI;
	  	double vidSURFtheta_y2 = atan2(-vidSURFrotMatNew.at<double>(2,0),sqrt(vidSURFrotMatNew.at<double>(2,1)*vidSURFrotMatNew.at<double>(2,1) + vidSURFrotMatNew.at<double>(2,2)*vidSURFrotMatNew.at<double>(2,2)))*180/CV_PI;
		double vidSURFtheta_z2 = atan2(vidSURFrotMatNew.at<double>(1,0),vidSURFrotMatNew.at<double>(0,0))*180/CV_PI;  
		
		//Check to find the correct rotation - if magnitude of x and z rot is bigger, then it is the wrong matrix. Also check if any angles larger than 100 (bigger than FOV) which means wrong matrix. THIS IS NOT PROVEN
		int VALID_ROT_MAT_FLAG = 0;
		if((abs(vidSURFtheta_x) > abs(vidSURFtheta_x2))&&(abs(vidSURFtheta_z) > abs(vidSURFtheta_z2)))
		  //cout << "****USE ANGLES 2" << endl;
		  VALID_ROT_MAT_FLAG = 2;
		else
		  VALID_ROT_MAT_FLAG = 1;
		if((abs(vidSURFtheta_x) > 100) || (abs(vidSURFtheta_y) > 100) || (abs(vidSURFtheta_z) > 100))
		  //cout << "****USE ANGLES 2 AGAIN" << endl;
		  VALID_ROT_MAT_FLAG += 20;
		else if((abs(vidSURFtheta_x2) > 100) || (abs(vidSURFtheta_y2) > 100) || (abs(vidSURFtheta_z2) > 100))
		  //cout << "****USE ANGLES 1" << endl;
		  VALID_ROT_MAT_FLAG += 10;

		//Choose between the two possible rotation matrices based on flag
		//double vidSURFanglesArr[3];

		if((VALID_ROT_MAT_FLAG == 1) || (VALID_ROT_MAT_FLAG == 11))
		  { 
		    vidSURFanglesArr[0] = vidSURFtheta_x;
		    vidSURFanglesArr[1] = vidSURFtheta_y;
		    vidSURFanglesArr[2] = vidSURFtheta_z;
		  }
		else if((VALID_ROT_MAT_FLAG == 2) || (VALID_ROT_MAT_FLAG == 22))
		  { 
		    vidSURFanglesArr[0] = vidSURFtheta_x2;
		    vidSURFanglesArr[1] = vidSURFtheta_y2;
		    vidSURFanglesArr[2] = vidSURFtheta_z2;
		  }		   

		Mat vidSURFangles = Mat(3,1,DataType<double>::type,vidSURFanglesArr);
		rotatedVidSURFangles = RcamtoV*Rneg30x*vidSURFangles;

		if(LAPTOP_CAMERA)
		  rotatedVidSURFangles = RcamtoV*Rneg20x*vidSURFangles;

		//Get the sum angles
		vidSURFsumAngles = vidSURFsumAngles + rotatedVidSURFangles;

		if(DISPLAY_POSE)
		  {
		    cout << "SURF VALID ROT MAT FLAG = " << VALID_ROT_MAT_FLAG << endl;
		    //cout << "vidSURFRot = " << vidSURFrotMat << endl;
		    cout << "vidSURFangles = " << vidSURFangles << endl;
		    cout << "rotatedVidSURFangles = " << rotatedVidSURFangles << endl;
		  }

		//cout << "Rotation: " << vidSURFrotMat << endl;
		/*		cout << "Video SURF Translation: " << vidSURFtranslMat << endl;
		cout << "Video SURF Thetax: " << vidSURFtheta_x << endl;
		cout << "Video SURF Thetay: " << vidSURFtheta_y << endl;
		cout << "Video SURF Thetaz: " << vidSURFtheta_z << endl;
		*/
		
		//cout << "SURF FUND MAT SIZE = " << vidSURFfundamentalMatrix << endl;
		/*
		if(vidSURFfundamentalMatrix.rows > 0)
		  {
		    if(vidSURFfundamentalMatrix.rows > 3)
                      {
                        vidSURFfundamentalMatrix.resize(3);
                        //cout << vidSURFfundamentalMatrix << endl;
		      }
		    //cout << "SURF fundmatrows = " << vidSURFfundamentalMatrix.rows << endl;
		    Mat vidSURFessentialMatrix = cameraMatrix_transpose*vidSURFfundamentalMatrix*cameraMatrix;
		    vidSURFmodel = vidSURFessentialMatrix.clone();
		    vidSURFnumInliers = sum(vidSURFmask)[0]; //Find number of inliers
		  }
		else
		  {
		    vidSURFmodel = Mat::zeros(3,3,CV_64F);
		    vidSURFmask = Mat::zeros(1,vidSURFgood_match_pts2.size(),CV_8U);
		    vidSURFnumInliers = 0;
		    //cout << "NO SURF FUNDAMENTAL MATRIX FOUND" << endl;
		  }
		*/

		//Draw Features: 
		drawMatches(prev_videoImgGray, vidSURFkeypoints_1, videoImgGray, vidSURFkeypoints_2, vidSURFgood_matches, vidMatchesImg,Scalar::all(-1), Scalar::all(-1), vector<char>()); //, DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		for(int i1 = 0; i1 < vidSURFgood_match_pts2.size(); i1++)
		  {
		    Scalar color;
		    if(vidSURFmask.at<char>(0,i1))
		      color = Scalar(0,255,0);
		    else
		      color = Scalar(0,0,255);
		    circle(vidFeaturesImg, vidSURFgood_match_pts1[i1], 4, color);
		  }
	      }

	    else
	      vidSURFmodel = Mat::zeros(3,3,CV_64F);

	  }
	else
	  vidSURFmodel = Mat::zeros(3,3,CV_64F);

	//cout << "Num SURF Matches: " << vidSURFnumMatches << endl;
	//cout << "Num SURF Inliers: " << vidSURFnumInliers << endl;
	//cout << "video SURF Model" << vidSURFmodel << endl;
	putText(vidFeaturesImg, "SURF", Point(10,25),FONT_HERSHEY_SIMPLEX,1,Scalar(255,255,255));
	putText(vidMatchesImg, "SURF", Point(10,25),FONT_HERSHEY_SIMPLEX,1,Scalar(255,255,255));
	if(DISPLAY_IMGS)
	  {
	    imshow(vid_features_wnd, vidFeaturesImg);
	    imshow(vid_matches_wnd, vidMatchesImg);
	    waitKey(0);
	    
	  }

	//---------------video SIFT Feature Detection ------------------//
	cvtColor(prev_videoImgGray, vidFeaturesImg, CV_GRAY2BGR); //Get copy of gray img to mark features
	vidMatchesImg = prev_videoImgGray.clone();
        Mat vidSIFTmodel, vidSIFTmask;
	float vidSIFTnumInliers = 0;
	float vidSIFTnumMatches = 0;
	double vidSIFTtheta_x, vidSIFTtheta_y, vidSIFTtheta_z;
	Mat vidSIFTrotMat;
	Mat vidSIFTtranslMat;
	double vidSIFTanglesArr[3] = {0,0,0};
	Mat rotatedVidSIFTangles = Mat::zeros(3,1,DataType<double>::type);

	std::vector<KeyPoint> vidSIFTkeypoints_1, vidSIFTkeypoints_2;
        vidSIFTdetector->detect( prev_videoImgGray, vidSIFTkeypoints_1 );
        vidSIFTdetector->detect( videoImgGray, vidSIFTkeypoints_2 );
        //SURF Calculate descriptors (feature vectors):
        Mat vidSIFTdescriptors_1, vidSIFTdescriptors_2;
        vidSIFTextractor->compute( prev_videoImgGray, vidSIFTkeypoints_1, vidSIFTdescriptors_1 );
        vidSIFTextractor->compute( videoImgGray, vidSIFTkeypoints_2, vidSIFTdescriptors_2 );

	if((!vidSIFTdescriptors_1.empty()) && (!vidSIFTdescriptors_2.empty()))
	  {
	    //SIFT Matching descriptor vectors using FLANN matcher
	    std::vector< DMatch > vidSIFTmatches;
	    vidSIFTmatcher.match( vidSIFTdescriptors_1, vidSIFTdescriptors_2, vidSIFTmatches );
	    double vidSIFTmax_dist = 0; double vidSIFTmin_dist = 100;
	    //-- Quick calculation of max and min distances between keypoints
	    for( int i1 = 0; i1 < vidSIFTdescriptors_1.rows; i1++ )
	      { double dist = vidSIFTmatches[i1].distance;
		if( dist < vidSIFTmin_dist ) vidSIFTmin_dist = dist;
		if( dist > vidSIFTmax_dist ) vidSIFTmax_dist = dist;
	      }
	    //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist, or a small arbitary value ( 0.02 ) in the event that min_dist is very small) //-- PS.- radiusMatch can also be used here.
	    std::vector< DMatch > vidSIFTgood_matches;
	    for( int i1 = 0; i1 < vidSIFTdescriptors_1.rows; i1++ )
	      { //if( vidSIFTmatches[i1].distance <= max(2*vidSIFTmin_dist, 0.02) )
		  { vidSIFTgood_matches.push_back( vidSIFTmatches[i1]); }
	      }
	    std::vector<Point2f> vidSIFTgood_match_pts1;
	    std::vector<Point2f> vidSIFTgood_match_pts2;
	    std::vector<Point2f> vidSIFTgood_match_pts_diff;
	    
	    
	    for( int i1 = 0; i1 < vidSIFTgood_matches.size(); i1++ )
	      {
		//-- Get the keypoints from the good matches
		vidSIFTgood_match_pts1.push_back( vidSIFTkeypoints_1[ vidSIFTgood_matches[i1].queryIdx ].pt );
		vidSIFTgood_match_pts2.push_back( vidSIFTkeypoints_2[ vidSIFTgood_matches[i1].trainIdx ].pt );
		vidSIFTgood_match_pts_diff.push_back( vidSIFTkeypoints_2[ vidSIFTgood_matches[i1].trainIdx].pt - vidSIFTkeypoints_1[ vidSIFTgood_matches[i1].queryIdx].pt);
	      }
	    vidSIFTnumMatches = vidSIFTgood_match_pts1.size();
	    
	    for(int i1 = 0; i1 < vidSIFTgood_match_pts1.size(); i1++)
	      {
		//cout << vidSIFTgood_match_pts2[i1] << "-" << vidSIFTgood_match_pts1[i1] << "=" << vidSIFTgood_match_pts_diff[i1] << endl;
	      }
	    
	    //Find Fundamental and Essential Matrices:
	    double vidSIFT_RANSAC_reprojthresh = 1;
	    double vidSIFT_RANSAC_param = 0.99;
	    
	    if(vidSIFTgood_matches.size() > 5)
	      {
		//Mat vidSIFTfundamentalMatrix = findFundamentalMat(vidSIFTgood_match_pts1, vidSIFTgood_match_pts2, ROBUST_EST_METH, vidSIFT_RANSAC_reprojthresh, vidSIFT_RANSAC_param,vidSIFTmask);
		
		Mat vidSIFTessentialMatrix = findEssentialMatNew(vidSIFTgood_match_pts1, vidSIFTgood_match_pts2, fx_new, fy_new, princ_pt, ROBUST_EST_METH, vidSIFT_RANSAC_param, vidSIFT_RANSAC_reprojthresh,vidSIFTmask); //ADD MASK LATER
		//cout << "Essential Mat 1 = " << vidSIFTessentialMatrix << endl;
		Mat tempMask = vidSIFTmask.clone();
		
		if(vidSIFTessentialMatrix.rows > 0)
		  {
		    //cout << "SIFT: " << vidSIFTessentialMatrix.rows/3 << "Essential Matrices Found" << endl;
		    if(vidSIFTessentialMatrix.rows > 3)
		      {	
			cout << "SIFT: " << vidSIFTessentialMatrix.rows/3 << "Essential Matrices Found" << endl;
			vidSIFTessentialMatrix.resize(3);
		      }
		    vidSIFTmodel = vidSIFTessentialMatrix.clone();
		    vidSIFTnumInliers = sum(vidSIFTmask)[0]; //Find number of inliers
		  }
		else
		  {
		    vidSIFTessentialMatrix = Mat::zeros(3,3,CV_64F);
		    vidSIFTmodel = Mat::zeros(3,3,CV_64F);
		    vidSIFTmask = Mat::zeros(1,vidSIFTgood_match_pts2.size(),CV_8U);
		    vidSIFTnumInliers = 0;
		    cout << "NO SIFT FUNDAMENTAL MATRIX FOUND" << endl;
		  }
		
		Mat tmpR1,tmpR2,tmpT;
		decomposeEssentialMat(vidSIFTessentialMatrix,tmpR1, tmpR2, tmpT);
		//cout << "SIFT TMP R1 = " << endl << tmpR1 << endl;
		//cout << "SIFT TMP R2 = " << endl << tmpR2 << endl;
		//cout << "SIFT TMP T = " << endl << tmpT << endl;
		
		recoverPoseNew(vidSIFTessentialMatrix, vidSIFTgood_match_pts1, vidSIFTgood_match_pts2, vidSIFTrotMat, vidSIFTtranslMat, fx_new, fy_new, princ_pt, tempMask);
		
		Mat vidSIFTrotMat = tmpR1;
		Mat vidSIFTrotMatNew = tmpR2;
		
		vidSIFTtheta_x = atan2(vidSIFTrotMat.at<double>(2,1),vidSIFTrotMat.at<double>(2,2))*180/CV_PI;
		vidSIFTtheta_y = atan2(-vidSIFTrotMat.at<double>(2,0),sqrt(vidSIFTrotMat.at<double>(2,1)*vidSIFTrotMat.at<double>(2,1) + vidSIFTrotMat.at<double>(2,2)*vidSIFTrotMat.at<double>(2,2)))*180/CV_PI;
		vidSIFTtheta_z = atan2(vidSIFTrotMat.at<double>(1,0),vidSIFTrotMat.at<double>(0,0))*180/CV_PI;
		
		double vidSIFTtheta_x2 = atan2(vidSIFTrotMatNew.at<double>(2,1),vidSIFTrotMatNew.at<double>(2,2))*180/CV_PI;
		double vidSIFTtheta_y2 = atan2(-vidSIFTrotMatNew.at<double>(2,0),sqrt(vidSIFTrotMatNew.at<double>(2,1)*vidSIFTrotMatNew.at<double>(2,1) + vidSIFTrotMatNew.at<double>(2,2)*vidSIFTrotMatNew.at<double>(2,2)))*180/CV_PI;
		double vidSIFTtheta_z2 = atan2(vidSIFTrotMatNew.at<double>(1,0),vidSIFTrotMatNew.at<double>(0,0))*180/CV_PI;
		
		//Check to find the correct rotation - if magnitude of x and z rot is bigger, then it is the wrong matrix. Also check if any angles larger than 100 (bigger than FOV) which means wrong matrix. THIS IS NOT PROVEN
		int VALID_ROT_MAT_FLAG = 0;
		if((abs(vidSIFTtheta_x) > abs(vidSIFTtheta_x2))&&(abs(vidSIFTtheta_z) > abs(vidSIFTtheta_z2)))
		  //cout << "****USE ANGLES 2" << endl;
		  VALID_ROT_MAT_FLAG = 2;
		else
		  VALID_ROT_MAT_FLAG = 1;
		if((abs(vidSIFTtheta_x) > 100) || (abs(vidSIFTtheta_y) > 100) || (abs(vidSIFTtheta_z) > 100))
		  //cout << "****USE ANGLES 2 AGAIN" << endl;
		  VALID_ROT_MAT_FLAG += 20;
		else if((abs(vidSIFTtheta_x2) > 100) || (abs(vidSIFTtheta_y2) > 100) || (abs(vidSIFTtheta_z2) > 100))
		  //cout << "****USE ANGLES 1" << endl;
		  VALID_ROT_MAT_FLAG += 10;
		
		//Choose between the two possible rotation matrices based on flag
		if((VALID_ROT_MAT_FLAG == 1) || (VALID_ROT_MAT_FLAG == 11))
		  {
		    vidSIFTanglesArr[0] = vidSIFTtheta_x;
		    vidSIFTanglesArr[1] = vidSIFTtheta_y;
		    vidSIFTanglesArr[2] = vidSIFTtheta_z;
		  }
		else if((VALID_ROT_MAT_FLAG == 2) || (VALID_ROT_MAT_FLAG == 22))
		  {
		    vidSIFTanglesArr[0] = vidSIFTtheta_x2;
		    vidSIFTanglesArr[1] = vidSIFTtheta_y2;
		    vidSIFTanglesArr[2] = vidSIFTtheta_z2;
		  }	
		
		Mat vidSIFTangles = Mat(3,1,DataType<double>::type,vidSIFTanglesArr);
		rotatedVidSIFTangles = RcamtoV*Rneg30x*vidSIFTangles;

		if(LAPTOP_CAMERA)
		  rotatedVidSIFTangles = RcamtoV*Rneg20x*vidSIFTangles;

		//Get the sum angles
		vidSIFTsumAngles = vidSIFTsumAngles + rotatedVidSIFTangles;
	
		if(DISPLAY_POSE)
		  {
		    cout << "SIFT VALID ROT MAT FLAG = " << VALID_ROT_MAT_FLAG << endl;
		    //cout << "vidSIFTRot = " << vidSIFTrotMat << endl;
		    cout << "vidSIFTangles = " << vidSIFTangles << endl;
		    cout << "rotatedVidSIFTangles = " << rotatedVidSIFTangles << endl;
		  }

		/* cout << "Rotation: " << vidSIFTrotMat << endl;
		   cout << "Video SIFT Translation: " << vidSIFTtranslMat << endl;
		   cout << "Video SIFT Thetax: " << vidSIFTtheta_x << endl;
		   cout << "Video SIFT Thetay: " << vidSIFTtheta_y << endl;
		   cout << "Video SIFT Thetaz: " << vidSIFTtheta_z << endl;
		*/
		
		/*
		//cout << vidSIFTfundamentalMatrix << endl;
		if(vidSIFTfundamentalMatrix.rows > 0)
		{
		if(vidSIFTfundamentalMatrix.rows > 3)
		{
		vidSIFTfundamentalMatrix.resize(3);
		//cout << "FMAT RESIZE: " << vidSIFTfundamentalMatrix << endl;
		}
		//cout << "SIFT fundmatrows = " << vidSIFTfundamentalMatrix.rows << endl;
		Mat vidSIFTessentialMatrix = cameraMatrix_transpose*vidSIFTfundamentalMatrix*cameraMatrix;
		vidSIFTmodel = vidSIFTessentialMatrix.clone();
		vidSIFTnumInliers = sum(vidSIFTmask)[0]; //Find number of inliers
		}
		else
		{
		vidSIFTmodel = Mat::zeros(3,3,CV_64F);
		vidSIFTmask = Mat::zeros(1,vidSIFTgood_match_pts2.size(),CV_8U);
		vidSIFTnumInliers = 0;
		//cout << "NO SIFT FUND MAT FOUND" << endl;
		}
		*/	
		
		//Draw Features:
		drawMatches(prev_videoImgGray, vidSIFTkeypoints_1, videoImgGray, vidSIFTkeypoints_2, vidSIFTgood_matches, vidMatchesImg,Scalar::all(-1), Scalar::all(-1), vector<char>()); //, DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		for(int i1 = 0; i1 < vidSIFTgood_match_pts1.size(); i1++)
		  {
		    Scalar color;
		    if(vidSIFTmask.at<char>(0,i1))
		      color = Scalar(0,255,0);
		    else
		      color = Scalar(0,0,255);
		    circle(vidFeaturesImg, vidSIFTgood_match_pts1[i1], 4, color);
		  }
		
	      }
	    else
	      vidSIFTmodel = Mat::zeros(3,3,CV_64F);
	    
	  }
	else
	  vidSIFTmodel = Mat::zeros(3,3,CV_64F);
	
	//cout << "Num SIFT Matches: " << vidSIFTnumMatches << endl;
	//cout << "Num SIFT Inliers: " << vidSIFTnumInliers << endl;
	//cout << "video SIFT model" << vidSIFTmodel << endl;
	putText(vidFeaturesImg, "SIFT", Point(10,25),FONT_HERSHEY_SIMPLEX,1,Scalar(255,255,255));
	putText(vidMatchesImg, "SIFT", Point(10,25),FONT_HERSHEY_SIMPLEX,1,Scalar(255,255,255));
	if(DISPLAY_IMGS)
	  {
	    imshow(vid_features_wnd, vidFeaturesImg);
	    imshow(vid_matches_wnd, vidMatchesImg);
	    waitKey(0);
	    imwrite("VidFeaturesIMG.jpg", vidFeaturesImg);
	    imwrite("VidMatchesIMG.jpg", vidMatchesImg);
	  }

	//-------------------------- HARRIS/SIFT Detection ---------------//
	//-- create detector and descriptor --
	// if you want it faster, take e.g. FAST or ORB
	Mat vidHARRISmodel, vidHARRISmask;
	float vidHARRISnumMatches = 0;
	float vidHARRISnumInliers = 0;
	double vidHARRIStheta_x, vidHARRIStheta_y, vidHARRIStheta_z;
	Mat vidHARRISrotMat;
	Mat vidHARRIStranslMat;
	double vidHARRISanglesArr[3] = {0,0,0};
	Mat rotatedVidHARRISangles = Mat::zeros(3,1,DataType<double>::type);

	cvtColor(prev_videoImgGray, vidFeaturesImg, CV_GRAY2BGR); //Get copy of gray img to mark features
	vidMatchesImg = prev_videoImgGray.clone();

	cv::Ptr<cv::FeatureDetector> vidHARRISdetector = cv::FeatureDetector::create("HARRIS"); 
	vidHARRISdetector->set("nfeatures", 100);
	//vidHARRISdetector->set("minDistance", 10);
	vidHARRISdetector->set("qualityLevel", 0.001); //0.01

	cv::Ptr<cv::DescriptorExtractor> vidHARRISdescriptor = cv::DescriptorExtractor::create("SIFT"); 

	// detect keypoints
	std::vector<cv::KeyPoint> vidHARRISkeypoints1, vidHARRISkeypoints2;
	vidHARRISdetector->detect(prev_videoImgGray, vidHARRISkeypoints1);
	vidHARRISdetector->detect(videoImgGray, vidHARRISkeypoints2);

	// extract features
	cv::Mat vidHARRISdesc1, vidHARRISdesc2;
	vidHARRISdescriptor->compute(prev_videoImgGray, vidHARRISkeypoints1, vidHARRISdesc1);
	vidHARRISdescriptor->compute(videoImgGray, vidHARRISkeypoints2, vidHARRISdesc2);

	if((!vidHARRISdesc1.empty()) && (!vidHARRISdesc2.empty()))
	  {
	    //HARRIS Matching descriptor vectors using FLANN matcher 
	    FlannBasedMatcher vidHARRISmatcher;
	    std::vector< DMatch > vidHARRISmatches;
	    vidHARRISmatcher.match( vidHARRISdesc1, vidHARRISdesc2, vidHARRISmatches );
	    double vidHARRISmax_dist = 0; double vidHARRISmin_dist = 100;
	    //-- Quick calculation of max and min distances between keypoints
	    for( int i1 = 0; i1 < vidHARRISdesc1.rows; i1++ )
	      { double dist = vidHARRISmatches[i1].distance;
		if( dist < vidHARRISmin_dist ) vidHARRISmin_dist = dist;
		if( dist > vidHARRISmax_dist ) vidHARRISmax_dist = dist;
	      }
	    //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist, 
	    //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very  
	    //-- small)   //-- PS.- radiusMatch can also be used here.                  
	    std::vector< DMatch > vidHARRISgood_matches;
	    for( int i1 = 0; i1 < vidHARRISdesc1.rows; i1++ )
	      { //if( vidHARRISmatches[i1].distance <= max(2*vidHARRISmin_dist, 0.02) )
		  { vidHARRISgood_matches.push_back( vidHARRISmatches[i1]); }
	      }
	    std::vector<Point2f> vidHARRISgood_match_pts1;
	    std::vector<Point2f> vidHARRISgood_match_pts2;
	    std::vector<Point2f> vidHARRISgood_match_pts_diff;
	        
	        
	    for( int i1 = 0; i1 < vidHARRISgood_matches.size(); i1++ )
	      {
		//-- Get the keypoints from the good matches                            
		vidHARRISgood_match_pts1.push_back( vidHARRISkeypoints1[ vidHARRISgood_matches[i1].queryIdx ].pt );
		vidHARRISgood_match_pts2.push_back( vidHARRISkeypoints2[ vidHARRISgood_matches[i1].trainIdx ].pt );
		vidHARRISgood_match_pts_diff.push_back(vidHARRISkeypoints2[vidHARRISgood_matches[i1].trainIdx].pt - vidHARRISkeypoints1[vidHARRISgood_matches[i1].queryIdx].pt);
	      }
	    vidHARRISnumMatches = vidHARRISgood_match_pts1.size();

	    for(int i1 = 0; i1 < vidHARRISgood_match_pts1.size(); i1++)
	      {
		//cout << vidHARRISgood_match_pts2[i1] << "-" << vidHARRISgood_match_pts1[i1] << "=" << vidHARRISgood_match_pts_diff[i1] << endl;
	      }


	    //Find Fundamental and Essential Matrices                    
	    double vidHARRIS_RANSAC_reprojthresh = 1;
	    double vidHARRIS_RANSAC_param = 0.99;
	   
	    if(vidHARRISgood_matches.size() > 5)
	      {
//		Mat vidHARRISfundamentalMatrix = findFundamentalMat(vidHARRISgood_match_pts1, vidHARRISgood_match_pts2, ROBUST_EST_METH, vidHARRIS_RANSAC_reprojthresh, vidHARRIS_RANSAC_param,vidHARRISmask);
	Mat vidHARRISessentialMatrix = findEssentialMatNew(vidHARRISgood_match_pts1, vidHARRISgood_match_pts2, fx_new, fy_new, princ_pt, ROBUST_EST_METH, vidHARRIS_RANSAC_param, vidHARRIS_RANSAC_reprojthresh,vidHARRISmask); //ADD MASK LATER
		//cout << "Essential Mat 1 = " << vidHARRISessentialMatrix1 << endl;
	Mat tempMask = vidHARRISmask.clone();

		if(vidHARRISessentialMatrix.rows > 0)
		  {
		    if(vidHARRISessentialMatrix.rows > 3)
		      vidHARRISessentialMatrix.resize(3);
		    vidHARRISmodel = vidHARRISessentialMatrix.clone();
		    vidHARRISnumInliers = sum(vidHARRISmask)[0]; //Find number of inliers		
		  }
		else
		  {
		    vidHARRISessentialMatrix = Mat::zeros(3,3,CV_64F);
		    vidHARRISmodel = Mat::zeros(3,3,CV_64F);
		    vidHARRISmask = Mat::zeros(1,vidHARRISgood_match_pts2.size(),CV_8U);
		    vidHARRISnumInliers = 0;
		    cout << "NO HARRIS FUNDAMENTAL MATRIX FOUND" << endl;
		  }

		Mat tmpR1,tmpR2,tmpT;
		decomposeEssentialMat(vidHARRISessentialMatrix,tmpR1, tmpR2, tmpT);
		//cout << "HARRIS TMP R1 = " << endl << tmpR1 << endl;
		//cout << "HARRIS TMP R2 = " << endl << tmpR2 << endl;
		//cout << "HARRIS TMP T = " << endl << tmpT << endl;

		recoverPoseNew(vidHARRISessentialMatrix, vidHARRISgood_match_pts1, vidHARRISgood_match_pts2, vidHARRISrotMat, vidHARRIStranslMat, fx_new, fy_new, princ_pt, tempMask);

		Mat vidHARRISrotMat = tmpR1;
		Mat vidHARRISrotMatNew = tmpR2;
		
		vidHARRIStheta_x = atan2(vidHARRISrotMat.at<double>(2,1),vidHARRISrotMat.at<double>(2,2))*180/CV_PI;
		vidHARRIStheta_y = atan2(-vidHARRISrotMat.at<double>(2,0),sqrt(vidHARRISrotMat.at<double>(2,1)*vidHARRISrotMat.at<double>(2,1) + vidHARRISrotMat.at<double>(2,2)*vidHARRISrotMat.at<double>(2,2)))*180/CV_PI;
		vidHARRIStheta_z = atan2(vidHARRISrotMat.at<double>(1,0),vidHARRISrotMat.at<double>(0,0))*180/CV_PI;  
		
		double vidHARRIStheta_x2 = atan2(vidHARRISrotMatNew.at<double>(2,1),vidHARRISrotMatNew.at<double>(2,2))*180/CV_PI;
	  	double vidHARRIStheta_y2 = atan2(-vidHARRISrotMatNew.at<double>(2,0),sqrt(vidHARRISrotMatNew.at<double>(2,1)*vidHARRISrotMatNew.at<double>(2,1) + vidHARRISrotMatNew.at<double>(2,2)*vidHARRISrotMatNew.at<double>(2,2)))*180/CV_PI;
		double vidHARRIStheta_z2 = atan2(vidHARRISrotMatNew.at<double>(1,0),vidHARRISrotMatNew.at<double>(0,0))*180/CV_PI;  
		
		//Check to find the correct rotation - if magnitude of x and z rot is bigger, then it is the wrong matrix. Also check if any angles larger than 100 (bigger than FOV) which means wrong matrix. THIS IS NOT PROVEN
		int VALID_ROT_MAT_FLAG = 0;
		if((abs(vidHARRIStheta_x) > abs(vidHARRIStheta_x2))&&(abs(vidHARRIStheta_z) > abs(vidHARRIStheta_z2)))
		  //cout << "****USE ANGLES 2" << endl;
		  VALID_ROT_MAT_FLAG = 2;
		else
		  VALID_ROT_MAT_FLAG = 1;
		if((abs(vidHARRIStheta_x) > 100) || (abs(vidHARRIStheta_y) > 100) || (abs(vidHARRIStheta_z) > 100))
		  //cout << "****USE ANGLES 2 AGAIN" << endl;
		  VALID_ROT_MAT_FLAG += 20;
		else if((abs(vidHARRIStheta_x2) > 100) || (abs(vidHARRIStheta_y2) > 100) || (abs(vidHARRIStheta_z2) > 100))
		  //cout << "****USE ANGLES 1" << endl;
		  VALID_ROT_MAT_FLAG += 10;

		//Choose between the two possible rotation matrices based on flag

		if((VALID_ROT_MAT_FLAG == 1) || (VALID_ROT_MAT_FLAG == 11))
		  { 
		    vidHARRISanglesArr[0] = vidHARRIStheta_x;
		    vidHARRISanglesArr[1] = vidHARRIStheta_y;
		    vidHARRISanglesArr[2] = vidHARRIStheta_z;
		  }
		else if((VALID_ROT_MAT_FLAG == 2) || (VALID_ROT_MAT_FLAG == 22))
		  { 
		    vidHARRISanglesArr[0] = vidHARRIStheta_x2;
		    vidHARRISanglesArr[1] = vidHARRIStheta_y2;
		    vidHARRISanglesArr[2] = vidHARRIStheta_z2;
		  }		   

		Mat vidHARRISangles = Mat(3,1,DataType<double>::type,vidHARRISanglesArr);
		rotatedVidHARRISangles = RcamtoV*Rneg30x*vidHARRISangles;

		if(LAPTOP_CAMERA)
		  rotatedVidHARRISangles = RcamtoV*Rneg20x*vidHARRISangles;

		//Get the sum angles
		vidHARRISsumAngles = vidHARRISsumAngles + rotatedVidHARRISangles;

		if(DISPLAY_POSE)
		  {
		    cout << "HARRIS VALID ROT MAT FLAG = " << VALID_ROT_MAT_FLAG << endl;
		    //cout << "vidHARRISRot = " << vidHARRISrotMat << endl;
		    cout << "vidHARRISangles = " << vidHARRISangles << endl;
		    cout << "rotatedVidHARRISangles = " << rotatedVidHARRISangles << endl;
		  }

		//cout << "Rotation: " << vidHARRISrotMat << endl;
		//cout << "Video HARRIS Translation: " << vidHARRIStranslMat << endl;
		//		cout << "Video HARRIS Thetax: " << vidHARRIStheta_x << endl;
		//cout << "Video HARRIS Thetay: " << vidHARRIStheta_y << endl;
		//cout << "Video HARRIS Thetaz: " << vidHARRIStheta_z << endl;


		/*
		if(vidHARRISfundamentalMatrix.rows > 0)
		  {
		    if(vidHARRISfundamentalMatrix.rows > 3)
                      {
                        vidHARRISfundamentalMatrix.resize(3);
                        //cout << "FMAT RESIZE: " << vidHARRISfundamentalMatrix << endl;
                      }
		    //cout << "HARRIS fundmatrows = " << vidHARRISfundamentalMatrix.rows << endl;		  
		    Mat vidHARRISessentialMatrix = cameraMatrix_transpose*vidHARRISfundamentalMatrix*cameraMatrix;
		    vidHARRISmodel = vidHARRISessentialMatrix.clone();
		    vidHARRISnumInliers = sum(vidHARRISmask)[0]; //Find number of inliers
		  }
		else
		  {
		    vidHARRISmodel = Mat::zeros(3,3,CV_64F);
		    vidHARRISmask = Mat::zeros(1,vidHARRISgood_match_pts2.size(), CV_8U);
		    vidHARRISnumInliers = 0;
		    //cout << "NO HARRIS FUNDAMENTAL MATRIX FOUND" << endl;
		  }
*/

		//Draw Features:
		drawMatches(prev_videoImgGray, vidHARRISkeypoints1, videoImgGray, vidHARRISkeypoints2, vidHARRISgood_matches, vidMatchesImg,Scalar::all(-1), Scalar::all(-1), vector<char>()); //, DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		for(int i1 = 0; i1 < vidHARRISgood_match_pts1.size(); i1++)
		  {
		    Scalar color;
		    if(vidHARRISmask.at<char>(0,i1))
		      color = Scalar(0,255,0);
		    else
		      color = Scalar(0,0,255);
		    circle(vidFeaturesImg, vidHARRISgood_match_pts1[i1], 4, color);
		  }
	      }

	    else
	      vidHARRISmodel = Mat::zeros(3,3,CV_64F);


	  }
	else vidHARRISmodel = Mat::zeros(3,3,CV_64F);


	//cout << "Num HARRIS Matches: " << vidHARRISnumMatches << endl;
	//cout << "Num HARRIS Inliers: " << vidHARRISnumInliers << endl;
	//cout << "Video HARRIS Model: " << vidHARRISmodel << endl;
	putText(vidFeaturesImg, "HARRIS", Point(10,25),FONT_HERSHEY_SIMPLEX,1,Scalar(255,255,255));
	putText(vidMatchesImg, "HARRIS", Point(10,25),FONT_HERSHEY_SIMPLEX,1,Scalar(255,255,255));
	if(DISPLAY_IMGS)
	  {
	    imshow(vid_features_wnd, vidFeaturesImg);
	    imshow(vid_matches_wnd, vidMatchesImg);
	    waitKey(0);
	  }

	//Calculate ground truth compass heading change:
	float heading_delta = heading_arr[prev_i] - heading_arr[i];	

	if(DISPLAY_POSE)
	  cout << "HEADING DELTA = " << heading_delta << endl;
	

	//--------------------------------------------------------//
	//----------------Print Results to File ------------------//
       	outfile << prev_i << ";" << i << ";" << heading_arr[prev_i] << ";" << heading_delta << ";";
	outfile << vidOFanglesArr[0] << ";" << vidOFanglesArr[1] << ";" << vidOFanglesArr[2] << ";" << rotatedVidOFangles.at<double>(0,0) << ";" << rotatedVidOFangles.at<double>(0,1) << ";" << rotatedVidOFangles.at<double>(0,2) << ";" << vidOFsumAngles.at<double>(0,0) << ";" << vidOFsumAngles.at<double>(0,1) << ";" << vidOFsumAngles.at<double>(0,2) << ";";
	outfile << vidSURFanglesArr[0] << ";" << vidSURFanglesArr[1] << ";" << vidSURFanglesArr[2] << ";" << rotatedVidSURFangles.at<double>(0,0) << ";" << rotatedVidSURFangles.at<double>(0,1) << ";" << rotatedVidSURFangles.at<double>(0,2) << ";" << vidSURFsumAngles.at<double>(0,0) << ";" << vidSURFsumAngles.at<double>(0,1) << ";" << vidSURFsumAngles.at<double>(0,2) << ";";
	outfile << vidSIFTanglesArr[0] << ";" << vidSIFTanglesArr[1] << ";" << vidSIFTanglesArr[2] << ";" << rotatedVidSIFTangles.at<double>(0,0) << ";" << rotatedVidSIFTangles.at<double>(0,1) << ";" << rotatedVidSIFTangles.at<double>(0,2) << ";" << vidSIFTsumAngles.at<double>(0,0) << ";" << vidSIFTsumAngles.at<double>(0,1) << ";" << vidSIFTsumAngles.at<double>(0,2) << ";";
	outfile << vidHARRISanglesArr[0] << ";" << vidHARRISanglesArr[1] << ";" << vidHARRISanglesArr[2] << ";" << rotatedVidHARRISangles.at<double>(0,0) << ";" << rotatedVidHARRISangles.at<double>(0,1) << ";" << rotatedVidHARRISangles.at<double>(0,2) << ";" << vidHARRISsumAngles.at<double>(0,0) << ";" << vidHARRISsumAngles.at<double>(0,1) << ";" << vidHARRISsumAngles.at<double>(0,2) << endl;

	//Output sonar data
	outfileSonar << prev_i << ";" << i << ";" << heading_arr[prev_i] << ";" << heading_delta << ";";
	outfileSonar << sonOFmodel.at<double>(0,0) << ";" << sonOFmodel.at<double>(0,1) << ";" << sonOFmodel.at<double>(0,2) << ";";
	outfileSonar << sonSURFmodel.at<double>(0,0) << ";" << sonSURFmodel.at<double>(0,1) << ";" << sonSURFmodel.at<double>(0,2) << ";";
	outfileSonar << sonSIFTmodel.at<double>(0,0) << ";" << sonSIFTmodel.at<double>(0,1) << ";" << sonSIFTmodel.at<double>(0,2) << ";";
	outfileSonar << sonHARRISmodel.at<double>(0,0) << ";" << sonHARRISmodel.at<double>(0,1) << ";" << sonHARRISmodel.at<double>(0,2) << endl;

        //Save Previous Images:
	if(DISPLAY_IMGS)
	  imshow(sonar_box_wnd, sonarImgGray);    
	prev_sonarImgGray = sonarImgGray.clone();
        prev_videoImgGray = videoImgGray.clone();
	prev_i = i;


	//--------------------------------------------------------//
	//--------------------------------------------------------//
	//--------------------------------------------------------//

	//-------------------------------------------------------
	//-------------Read in compass, depth readings
      	heading_deg = heading_arr[i];
	depth = depth_arr[i];

       	//cout << "Heading = " << heading_deg << "  Depth = " << depth << endl;

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

        if(WRITE_FRAME_NUM){
	  //Put Frame Number on image                                                  
	  sprintf(text, "%d", i);
	  cv::putText(colorImg, text, cvPoint(0,25), cv::FONT_HERSHEY_SIMPLEX,1,cv::Scalar::all(255));
        }

	//Display Images:
	if(DISPLAY_IMGS)
	  {
	    cv::imshow(color_wnd, colorImg);
	    //std::cout << frame_num_windows_int << std::endl;
	    cv::imshow(video_wnd, videoImg);
	    cv::imshow(video_undist_wnd, videoImgUndistort);
	    cv::imshow(compass_wnd, compassImg);
	    cv::imshow(depth_wnd, depthImg);
	    //imshow(son_features_wnd, sonFeaturesImg);
	    //imshow(vid_features_wnd, vidFeaturesImg);
	  }
	//print frame information	
	//printf("i=%d\n", i);
	if(i==1)
		printf("height: %d width: %d\n", height, width);

	  BVTPing_Destroy(ping);

	  
	//------------------------------------
	//Check for key press:
	if(pause)
	{
	  while(1){
		keypushed = waitKey(10);
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
		keypushed = waitKey(10);
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
	
     cvReleaseImageHeader(&color_img_ipl);
     cvReleaseImageHeader(&mag_img_ipl);

     destroyWindow(color_wnd);
     destroyWindow(video_wnd);
     destroyWindow(compass_wnd);
     destroyWindow(depth_wnd);
	
     // Clean up
     BVTColorImage_Destroy(bvtclr_img);
     BVTMagImage_Destroy(bvtmag_img);
     BVTColorMapper_Destroy(mapper);
     BVTSonar_Destroy(son);

     inFileCompass.close();
     outfile.close();
     outfileSonar.close();

     return 0;
}
