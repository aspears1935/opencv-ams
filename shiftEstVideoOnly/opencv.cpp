/*
 * Video 6DOF shift est Example using OpenCV
 */


//TO DO:
//GET VIDEO CHESS BOARDS FROM ICEFIN OR 720x480 imgs
//Don't have to set clahe clip limit?
//Matches inliers image too?
//Create a video of the results?
//Make sure directions are right for transl and rot
//Maybe change bad data from all zeros to the last good result?
//Add a max change in rotation/translation between frames to elimiated huge jumps?
//LATER: Faster if don't recompute corners for past image
//REVISIT LATER: How to deal with transl estimates in purely rotational situation. coupled. x and y see sinusoidal patterns, Z translation is mostly -1. - Maybe use Std Dev over last 10?
//MIGHT CONSIDER: Remove trackbar - doesn't really work - too much computation.
//OK FOR NOW (if need essentialMatrix later revisit. For now just use the euler angles) Check and make sure essentialMatZero works with all. And isn't needed much.
//OK FOR NOW (just do in excel). Think about doing an average and standard deviation for each. Output quantitiave results
//OK FOR NOW (Changed minHessian to 400 from 300. But look at timing and maybe get better solution.) Reduce number of SURF matches. gets up to 4530 with brash ice. SURF performs poorly on down lake ice circle, but well on Brash ice. Can't set number of feautres to detect, only minHessian - Maybe change minHessian depending on number detected, but this would take time too - Maybe don't use SURF? - Look at timing
//MULTIPLE E MATS: For now, ignore because it never happens. Might need to figure out way of dealing with multiple E mats if needed though.
//CHIRALITY: Need to do Chirality check on tranlsation vector. For now I'm using recoverPose which is redundant to decomposeEssentialMat, but could create custom function to speedup.


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
#include "RANSAC.hpp"
#include "my_ptsetreg.hpp"
#include "my_five-point.hpp"

using namespace std;
using namespace cv;

#define DISPLAY_IMGS 	false
#define STOP_BETWEEN_IMGS false
#define VERBOSE    false
#define WRITE_FRAME_NUM false
#define BLUR_VID_IMGS   false
#define WRITE_IMGS      false
#define PROCESS_00_FRAME false
#define OPENCV_ROTS_TO_CSV false
#define THRESHOLD_JUMPS true
#define ZERO_ROLLPITCH true
#define WRITE_VIDEO false

#define PI 3.14159265359
#define ZERO 0.0001
#define max_BINARY_value 255
#define MAX_ROT 30 //30 degrees is max frame-frame rotation

#define LINUX0DEG  0
#define LINUX17DEG 1
#define LINUX30DEG 2
#define LINUX68DEG 3
#define WINDOWS0DEG 4
#define WINDOWS17DEG 5
#define WINDOWS30DEG 6
#define WINDOWS68DEG 7
#define BLENDER_DOWN 8
#define BLENDER_FRONT 9
#define GTRI_LAPTOP 10
#define GOPRO_BRITNEY 11
#define ICEFIN_FRONT 12
#define ICEFIN_DOWN 13

#define MAPSAC 7
#define ROBUST_EST_METH CV_FM_LMEDS //CV_FM_LMEDS or CV_FM_RANSAC or MAPSAC

//------------Initialize the camera matrices---------------//
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
double camArrBlender[3][3] = {1050, 0, 479.5, 0, 1050, 269.5, 0, 0, 1};

double camArrGoProBritney[3][3] = {601.638, 0, 639.5, 0, 599.159, 479.5, 0, 0, 1};
//640 width:
//double camArrIcefinFront[3][3] = {811.499, 0, 319.5, 0, 806.21, 239.5 ,0, 0, 1};
//double camArrIcefinDown[3][3] = {315.625, 0, 319.5, 0, 414.976, 239.5, 0, 0, 1};
//720 width estimated (calculated):
double camArrIcefinFront[3][3] = {912.936, 0, 359.5, 0, 806.21, 239.5 ,0, 0, 1};
double camArrIcefinDown[3][3] = {355.078, 0, 359.5, 0, 414.976, 239.5, 0, 0, 1};

//------------Initialize the Distortion matrices-------------//
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
double distArrBlender[5][1] = 
  {0, 0, 0, 0, 0};
double distArrGoProBritney[5][1] = 
  {-0.268852, 0.0873101, -0.00095488, -0.00197115, -0.0137625};
double distArrIcefinFront[5][1] = 
  {-0.433162, 0.197389, -0.00084873, -0.00105264, 0.0779082}; //640 width
double distArrIcefinDown[5][1] = 
  {-0.323215, 0.127643, 0.0000811, -0.00435598, -0.0253347}; //640 width


 bool FRONT_CAMERA;
 int DES_FOVX = 80; //Desired FOV in X direction. 80 deg for VideoRay. Not used for Blender
 int camDataType; //Windows or Linux and angle of sensor - default BLENDER cam
//RNG rng(12345); //Not used right now.

//Essential Matrix for zero translation and rotation
double essentialMatZero[3][3] = {0, -0.40824829, -0.40824829, 0.40824829, 0, -0.40824829, 0.40824829, 0.40824829, 0}; //Essential Matrix for rotation and all equal rotation 

 //Variable to save trackbar frame number
 int i = 0;
 bool tbar_update = false;

//-------------Trackbar Handler------------------//
 void Trackbar(int, void*)
 {
   tbar_update = true;
 }

//-------------Main Function------------------//
 int main( int argc, char *argv[] )
 {

 	bool pause = false; 
 	char keypushed = 0;

	int OF_EstValid = -1;
	int SURF_EstValid = -1;
	int SIFT_EstValid = -1;
	int HARRIS_EstValid = -1;

	//Initialize the sum angles
	Mat vidOFsumAngles = Mat::zeros(3,1,DataType<double>::type);
	Mat vidSURFsumAngles = Mat::zeros(3,1,DataType<double>::type);
	Mat vidSIFTsumAngles = Mat::zeros(3,1,DataType<double>::type);
	Mat vidHARRISsumAngles = Mat::zeros(3,1,DataType<double>::type);

	double xinit = 0;
	double yinit = 0;
	double zinit = 0;
	double rollinit = 0;
	double pitchinit = 0;
	double yawinit = 0;
	
	cout << "Using OpenCV v" << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << "." << CV_SUBMINOR_VERSION << endl;

      if (argc < 3) {
 	  printf("usage: ./sonar_opencv <video-file> <camera-type> <OPTIONAL xinit> <OPTIONAL yinit> <OPTIONAL zinit> <OPTIONAL rollinit> <OPTIONAL pitchinit> <OPTIONAL yawinit>\n");
 	  printf("example: ./sonar_opencv ../../../data/swimmer.avi 8\n");
 	  printf("example: ./sonar_opencv ../../../data/swimmer.avi 8 0 0 0 0 0 90\n");
	  cout << endl << "Camera Types:" << endl;
	  cout << "0 - LINUX0DEG" << endl;
	  cout << "1 - LINUX17DEG" << endl;
	  cout << "2 - LINUX30DEG" << endl;
	  cout << "3 - LINUX68DEG" << endl;
	  cout << "4 - WINDOWS0DEG" << endl;
	  cout << "5 - WINDOWS17DEG" << endl;
	  cout << "6 - WINDOWS30DEG" << endl;
	  cout << "7 - WINDOWS68DEG" << endl;
	  cout << "8 - BLENDER DOWN" << endl;
	  cout << "9 - BLENDER FRONT" << endl;
	  cout << "10 - GTRI LAPTOP" << endl;
	  cout << "11 - GOPRO BRITNEY" << endl;
	  cout << "12 - ICEFIN FRONT" << endl;
	  cout << "13 - ICEFIN DOWN" << endl;
 	  exit(-1);
      }

      char videoFileName[256];
      strcpy(videoFileName, argv[1]);
      camDataType = (int)atoi(argv[2]);

      if(argc > 3)
	xinit = atof(argv[3]);
      if(argc > 4)
	yinit = atof(argv[4]);
      if(argc > 5)
	zinit = atof(argv[5]);
      if(argc > 6)
	rollinit = atof(argv[6]);
      if(argc > 7)
	pitchinit = atof(argv[7]);
      if(argc > 8)
	yawinit = atof(argv[8]);
      
      cout << "Initial Global Values: x,y,z,roll,pitch,yaw: " << xinit << "," << yinit << "," << zinit << "," << rollinit << "," << pitchinit << "," << yawinit << endl;
      
      //Initialize the global sum angles
      double initAngles[3] = {rollinit, pitchinit, yawinit};
      Mat vidOFsumAnglesGlobal = Mat(3,1,DataType<double>::type,initAngles);
      Mat vidSURFsumAnglesGlobal = vidOFsumAnglesGlobal.clone(); //Can't initialize all to same array - pointer, not copy data
      Mat vidSIFTsumAnglesGlobal = vidOFsumAnglesGlobal.clone();
      Mat vidHARRISsumAnglesGlobal = vidOFsumAnglesGlobal.clone();

      //Initialize the global sum x,y,z position
      double initTransl[3] = {xinit,yinit,zinit};
      Mat vidOFsumTranslGlobal = Mat(3,1,DataType<double>::type,initTransl);
      Mat vidSURFsumTranslGlobal = vidOFsumTranslGlobal.clone();
      Mat vidSIFTsumTranslGlobal = vidOFsumTranslGlobal.clone();
      Mat vidHARRISsumTranslGlobal = vidOFsumTranslGlobal.clone();

 	//Create Windows:
 	char video_wnd[] = "Video: 'b'=back, 'f'=forward, 'p'=pause, 'ESC'=exit";
	char video_undist_wnd[] = "Undistorted Video";
	char vid_features_wnd[] = "Video Features Window";
	char vid_matches_wnd[] = "Video Matches Window";
	if(DISPLAY_IMGS)
	  {
	    namedWindow(video_wnd,1);	
	    namedWindow(video_undist_wnd,1);
	    namedWindow(vid_features_wnd,1);
	    namedWindow(vid_matches_wnd,1);
	  }

	//-------------Open Input Video File---------------------
	cv::VideoCapture inputVideo(videoFileName);
	if(!inputVideo.isOpened())
	  {
	    std::cout << "Could not open the input video: " << videoFileName << std::endl;
	    return -1;
	  }

	//-------------Get Video Properties-------------------
	cv::Size videoSize = cv::Size((int) inputVideo.get(CAP_PROP_FRAME_WIDTH),
				      (int) inputVideo.get(CAP_PROP_FRAME_HEIGHT));
	double VIDEO_FPS = inputVideo.get(CAP_PROP_FPS);
	double VIDEO_FRAME_COUNT = inputVideo.get(CAP_PROP_FRAME_COUNT);
	int numVidFrames = (int)VIDEO_FRAME_COUNT;
	std::cout << "Input frame resolution: Width=" << videoSize.width << "  Height=" << videoSize.height << " Num of Frames=" << numVidFrames << " FPS=" << inputVideo.get(CAP_PROP_FPS) << std::endl;

	//------------------Create Mat Structures---------------------
	cv::Mat videoImg, vidFeaturesImg, vidMatchesImg;
	cv::Mat vidMatchesImgOF, vidMatchesImgSURF;

	//-------------------Create Trackbars-------------------------             
	int max_frame_num = numVidFrames;
        char position_tbar[] = "Position";
	if(DISPLAY_IMGS)
	  cv::createTrackbar(position_tbar, video_wnd, &i, max_frame_num, Trackbar);

	//-----------------------------------------------------------
	//------------------- Initialize Feature Detection ----------
	//-----------------------------------------------------------
	//SURF Detector:
	int minHessian = 400; //Suggested 300-500. Higher val gives less features. Was 300 but gave too many.
	SurfFeatureDetector vidSURFdetector(minHessian);
	SurfDescriptorExtractor vidSURFextractor;
	FlannBasedMatcher vidSURFmatcher;

	//SIFT Detector:
	cv::Ptr< cv::FeatureDetector > vidSIFTdetector = FeatureDetector::create("SIFT");
	cv::Ptr< cv::DescriptorExtractor > vidSIFTextractor = DescriptorExtractor::create("SIFT");
	double SIFTthreshold = 0.001; //0.04
	double SIFTedgeThreshold = 100; //10 -> larger = more features
	vidSIFTdetector->set("contrastThreshold", SIFTthreshold);
	vidSIFTdetector->set("edgeThreshold", SIFTedgeThreshold);
	vidSIFTdetector->set("nFeatures", 1000);
	vidSIFTdetector->set("nOctaveLayers", 5);
	vidSIFTdetector->set("sigma", 1.0);//1.6 lower seems to be better to a point
	FlannBasedMatcher vidSIFTmatcher;
	//BFMatcher vidSIFTmatcher; //Brute Force matcher

	cv::Ptr<cv::FeatureDetector> vidHARRISdetector = cv::FeatureDetector::create("HARRIS"); 
	cv::Ptr<cv::DescriptorExtractor> vidHARRISdescriptor = cv::DescriptorExtractor::create("SIFT"); 
	vidHARRISdetector->set("nfeatures", 1000);
	//vidHARRISdetector->set("minDistance", 10);
	vidHARRISdetector->set("qualityLevel", 0.001); //0.01
	FlannBasedMatcher vidHARRISmatcher;

	//DEBUG - create Harris Feature Detector and print out properties:
	/*cv::Ptr< cv::FeatureDetector > vidSIFTdetector1 = FeatureDetector::create("HARRIS");	
	std::vector<cv::String> parameters;
	vidSIFTdetector1->getParams(parameters);
	for(int i2 = 0; i2 < parameters.size(); i2++)
	  {
	    cout <<  parameters[i2] << endl;
	  }
	  for(;;);*/
	  //End DEBUG

	//DEBUG - create SIFT detector without common interface
	/*
	SiftFeatureDetector vidSIFTdetector(SIFTthreshold, SIFTedgeThreshold);
OA	SiftDescriptorExtractor vidSIFTextractor;
	*/

	//Create CLAHE:
	int CLAHEclipLimit = 4;//DEBUG should be 6
	cv::Size CLAHEtileGridSize(16,16);//was 16,16
	Ptr<CLAHE> clahe = createCLAHE(CLAHEclipLimit, CLAHEtileGridSize);
	//clahe->setClipLimit(clipLimit); //Set CLAHE clip limit, higher more effect

	//----------------------------------------------------------------------
	//--------------------Initialize Camera Detection ----------------------
	//----------------------------------------------------------------------

	//Get the camera matrix and distortion coeffs for correct angle and camera
	Mat cameraMatrix;
	Mat distCoeffs;
	switch(camDataType)
	  {
	  case 0:
	    cout << "CAMERA ANGLE = 0-DEG LINUX" << endl;
	    cameraMatrix = Mat(3,3,DataType<double>::type,camArr0degLinux);
	    distCoeffs = Mat(5,1,DataType<double>::type,distArr0degLinux);
	    FRONT_CAMERA = true;
	    break;
	  case 1:
	    cout << "CAMERA ANGLE = 17-DEG LINUX" << endl;
	    cameraMatrix = Mat(3,3,DataType<double>::type,camArr17degLinux);
	    distCoeffs = Mat(5,1,DataType<double>::type,distArr17degLinux);
	    FRONT_CAMERA = true;
	    break;
	  case 2:
	    cout << "CAMERA ANGLE = 30-DEG LINUX" << endl;
	    cameraMatrix = Mat(3,3,DataType<double>::type,camArr30degLinux);
	    distCoeffs = Mat(5,1,DataType<double>::type,distArr30degLinux);
	    FRONT_CAMERA = true;
	    break;
	  case 3:
	    cout << "CAMERA ANGLE = 68-DEG LINUX" << endl;
	    cameraMatrix = Mat(3,3,DataType<double>::type,camArr68degLinux);
	    distCoeffs = Mat(5,1,DataType<double>::type,distArr68degLinux);
	    FRONT_CAMERA = true;
	    break;
	  case 4:
	    cout << "CAMERA ANGLE = 0-DEG WINDOWS" << endl;
	    cameraMatrix = Mat(3,3,DataType<double>::type,camArr0degWin);
	    distCoeffs = Mat(5,1,DataType<double>::type,distArr0degWin);
	    FRONT_CAMERA = true;
	    break;
	  case 5:
	    cout << "CAMERA ANGLE = 17-DEG WINDOWS" << endl;
	    cameraMatrix = Mat(3,3,DataType<double>::type,camArr17degWin);
	    distCoeffs = Mat(5,1,DataType<double>::type,distArr17degWin);
	    FRONT_CAMERA = true;
	    break;
	  case 6:
	    cout << "CAMERA ANGLE = 30-DEG WINDOWS" << endl;
	    cameraMatrix = Mat(3,3,DataType<double>::type,camArr30degWin);
	    distCoeffs = Mat(5,1,DataType<double>::type,distArr30degWin);
	    FRONT_CAMERA = true;
	    break;
	  case 7:
	    cout << "CAMERA ANGLE = 68-DEG LINUX" << endl;
	    cameraMatrix = Mat(3,3,DataType<double>::type,camArr68degWin);
	    distCoeffs = Mat(5,1,DataType<double>::type,distArr68degWin);
	    FRONT_CAMERA = true;
	    break;
	  case 8:
	    cout << "CAMERA = BLENDER DOWN" << endl;
	    cameraMatrix = Mat(3,3,DataType<double>::type,camArrBlender);
	    distCoeffs = Mat(5,1,DataType<double>::type,distArrBlender);
	    FRONT_CAMERA = false;
	    break;
	  case 9:
	    cout << "CAMERA = BLENDER FRONT" << endl;
	    cameraMatrix = Mat(3,3,DataType<double>::type,camArrBlender);
	    distCoeffs = Mat(5,1,DataType<double>::type,distArrBlender);
	    FRONT_CAMERA = true;
	    break;
	  case 10:
	    cout << "CAMERA = GTRI LAPTOP" << endl;	    
	    cameraMatrix = Mat(3,3,DataType<double>::type,camArrLaptop);
	    distCoeffs = Mat(5,1,DataType<double>::type,distArrLaptop);
	    FRONT_CAMERA = true;
	    DES_FOVX = 50; // Reset the Desired FOV to 50 deg
	    break;
	  case 11:
	    cout << "CAMERA ANGLE = GoPro Britney" << endl;
	    cameraMatrix = Mat(3,3,DataType<double>::type,camArrGoProBritney);
	    distCoeffs = Mat(5,1,DataType<double>::type,distArrGoProBritney);
	    FRONT_CAMERA = true;
	    break;
	  case 12:
	    cout << "CAMERA ANGLE = Icefin Front" << endl;
	    cameraMatrix = Mat(3,3,DataType<double>::type,camArrIcefinFront);
	    distCoeffs = Mat(5,1,DataType<double>::type,distArrIcefinFront);
	    FRONT_CAMERA = true;
	    DES_FOVX = 45; // Reset the Desired FOV to 55 deg- 60 gets all
	    break;
	  case 13:
	    cout << "CAMERA ANGLE = Icefin Down" << endl;
	    cameraMatrix = Mat(3,3,DataType<double>::type,camArrIcefinDown);
	    distCoeffs = Mat(5,1,DataType<double>::type,distArrIcefinDown);
	    FRONT_CAMERA = false;
	    break;
	  default:
	    cout << "ERROR DETERMINING CAMERA + ANGLE. USING 30-DEG LINUX" << endl;
	    cameraMatrix = Mat(3,3,DataType<double>::type,camArr30degLinux);
	    distCoeffs = Mat(5,1,DataType<double>::type,distArr30degLinux);
	    FRONT_CAMERA = true;
	    break;
	  }

	Mat cameraMatrix_transpose;
	transpose(cameraMatrix,cameraMatrix_transpose);

	//Create new camera matrix with desired FOVx
	Mat newCameraMatrix = cameraMatrix.clone();
	double fx_old = cameraMatrix.at<double>(0,0);
	double fy_old = cameraMatrix.at<double>(1,1);
	double fx_new = (videoSize.width/2)/(tan((CV_PI/180)*((double)DES_FOVX)/2));
	double fy_new = (fx_new*fy_old)/fx_old;
	Point2d princ_pt = Point2d(cameraMatrix.at<double>(0,2),cameraMatrix.at<double>(1,2));
	newCameraMatrix.at<double>(0,0) = fx_new;
	newCameraMatrix.at<double>(1,1) = fy_new;

	if((camDataType == BLENDER_DOWN)||(camDataType == BLENDER_FRONT)) //If Blender No New Camera Matrix, no distortion
	  {
	    cout << "camDataType = Blender - No Distortion" << endl;
	    newCameraMatrix = cameraMatrix.clone();
	    fx_new = fx_old;
	    fy_new = fy_old;
	  }

	cout << "Old Cam Mat: " << cameraMatrix << endl;
	cout << "New Cam Mat: " << newCameraMatrix << endl;

	if((camDataType!=BLENDER_DOWN)&&(camDataType!=BLENDER_FRONT))
	  {
	    cout << "fx ratio: " << fx_old/fx_new << endl;
	    cout << "fy ratio: " << fy_old/fy_new << endl;
	  }
	cout << "princ_pt: " << princ_pt << endl;

	//Get undistortion rectify map:
	Mat map1, map2;
	//videoSize = Size(720,480); //DEBUG - set video size manually!!!
	cout << "VID SIZE: " << videoSize << endl;
	initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(), newCameraMatrix, videoSize, CV_32FC1, map1, map2);

	//-------------------------------------------------
	//Rotation matrices for rotating between camera and vehicle coordinates
	Mat RcamtoV;
	double camtoVehArrFwd[3][3] = {0, 0, 1,  //FRONT_CAMERA array[down][right]
				       1, 0, 0, 
				       0, 1, 0};
	//double camtoVehArrDown[3][3] = {0, -1, 0,  //DOWN_CAMERA
	//				1, 0, 0, 
	//				0, 0, 1};
	double camtoVehArrUp[3][3] = {0, -1, 0,  //UPWARD_CAMERA
				      -1, 0, 0, 
				      0, 0, -1};
	if(FRONT_CAMERA)
	  {
	    cout << "Front Camera to Vehicle Matrix:" << endl;
	    RcamtoV = Mat(3,3,DataType<double>::type,camtoVehArrFwd);
	  }
	else //UPWARD_CAMERA
	  {
	    cout << "Upward Camera to Vehicle Matrix:" << endl;
	    RcamtoV = Mat(3,3,DataType<double>::type,camtoVehArrUp);
	  }

	double neg17deg = -17*CV_PI/180;
	double neg20deg = -20*CV_PI/180;
	double neg30deg = -30*CV_PI/180;
	double neg68deg = -68*CV_PI/180;

	double rotZero[3][3] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
	double rotNeg17x[3][3] = {1, 0, 0, 0, cos(neg17deg), sin(neg17deg), 0, -sin(neg17deg), cos(neg17deg)};
	double rotNeg20x[3][3] = {1, 0, 0, 0, cos(neg20deg), sin(neg20deg), 0, -sin(neg20deg), cos(neg20deg)};
	double rotNeg30x[3][3] = {1, 0, 0, 0, cos(neg30deg), sin(neg30deg), 0, -sin(neg30deg), cos(neg30deg)};
	double rotNeg68x[3][3] = {1, 0, 0, 0, cos(neg68deg), sin(neg68deg), 0, -sin(neg68deg), cos(neg68deg)};

	Mat Rzero   = Mat(3,3,DataType<double>::type,rotZero);
	Mat Rneg17x = Mat(3,3,DataType<double>::type,rotNeg17x);
	Mat Rneg20x = Mat(3,3,DataType<double>::type,rotNeg20x);
	Mat Rneg30x = Mat(3,3,DataType<double>::type,rotNeg30x);
	Mat Rneg68x = Mat(3,3,DataType<double>::type,rotNeg68x);

	cout << "RcamtoV = " << endl << RcamtoV << endl;
	//	cout << "Rneg30x = " << endl << Rneg30x << endl;


	//Open Mask for Icefin Front
	Mat icefinFrontMask = imread("icefinFrontMask.png",IMREAD_GRAYSCALE);
	icefinFrontMask=icefinFrontMask/255; //convert to 0s and 1s

      //------Open Output Files, write column headings and run properties--------//
      //-------------------------------------------------------------------------//
      ofstream outfile("output.csv");
      outfile << "PROPERTIES:;";
      outfile << "VideoFileName=" << videoFileName << ";";
      outfile << "CameraType=" << camDataType << ";";
      outfile << "videoSize=" << videoSize << ";";
      outfile << "videoFPS=" << VIDEO_FPS << ";";
      outfile << "videoFrameCount=" << numVidFrames << ";";
      outfile << "SURFminHessian=" << minHessian << ";";
      outfile << "SIFTthreshold=" << SIFTthreshold << ";";
      outfile << "SIFTedgeThresh=" << SIFTedgeThreshold << ";";
      outfile << "CLAHEclipLimit=" << CLAHEclipLimit << ";";
      outfile << "CLAHEtileSize=" << CLAHEtileGridSize << ";";
      outfile << "fx_new=" << fx_new << ";";
      outfile << "fy_new=" << fy_new << ";";
      outfile << "princPt=" << princ_pt << ";";
      //outfile << "camDistCoeffs=" << distCoeffs << ";";
      outfile << "FRONT_CAMERA=" << FRONT_CAMERA << ";";
      //outfile << "camtoVehMat=" << RcamtoV << ";";
      outfile << "DES_FOVX=" << DES_FOVX << ";";
      outfile << "BLUR_IMGS=" << BLUR_VID_IMGS << ";";
      outfile << "ROBUST_EST=" << ROBUST_EST_METH << ";";
      outfile << "PROCESS_00FRAME=" << PROCESS_00_FRAME << ";";
      outfile << "THRESHOLD_JUMPS=" << THRESHOLD_JUMPS << ";";
      outfile << "MAX_ROT=" << MAX_ROT << ";";
      outfile << endl << endl;

      outfile << "frame1;frame2;";
      if(OPENCV_ROTS_TO_CSV)
	outfile << "vidOF_x;vidOF_y;vidOF_z;";
      outfile << "vidOF_x_rot;vidOF_y_rot;vidOF_z_rot;vidOF_x_sum;vidOF_y_sum;vidOF_z_sum;";
      if(OPENCV_ROTS_TO_CSV)
	outfile << "vidSURF_x;vidSURF_y;vidSURF_z;";
      outfile << "vidSURF_x_rot;vidSURF_y_rot;vidSURF_z_rot;vidSURF_x_sum;vidSURF_y_sum;vidSURF_z_sum;";
      if(OPENCV_ROTS_TO_CSV)
	outfile << "vidSIFT_x;vidSIFT_y;vidSIFT_z;";
      outfile << "vidSIFT_x_rot;vidSIFT_y_rot;vidSIFT_z_rot;vidSIFT_x_sum;vidSIFT_y_sum;vidSIFT_z_sum;";
      if(OPENCV_ROTS_TO_CSV)
	outfile << "vidHARRIS_x;vidHARRIS_y;vidHARRIS_z;";
      outfile << "vidHARRIS_x_rot;vidHARRIS_y_rot;vidHARRIS_z_rot;vidHARRIS_x_sum;vidHARRIS_y_sum;vidHARRIS_z_sum;";
      outfile << "vidOFtranslMat_x;vidOFtranslMat_y;vidOFtranslMat_z;vidSURFtranslMat_x;vidSURFtranslMat_y;vidSURFtranslMat_z;vidSIFTtranslMat_x;vidSIFTtranslMat_y;vidSIFTtranslMat_z;vidHARRIStranslMat_x;vidHARRIStranslMat_y;vidHARRIStranslMat_z;";

      outfile << "vidOFglobalSum_x;vidOFglobalSum_y;vidOFglobalSum_z;vidOFglobalSum_roll;vidOFglobalSum_pitch;vidOFglobalSum_yaw;";
      outfile << "vidSURFglobalSum_x;vidSURFglobalSum_y;vidSURFglobalSum_z;vidSURFglobalSum_roll;vidSURFglobalSum_pitch;vidSURFglobalSum_yaw;";
      outfile << "vidSIFTglobalSum_x;vidSIFTglobalSum_y;vidSIFTglobalSum_z;vidSIFTglobalSum_roll;vidSIFTglobalSum_pitch;vidSIFTglobalSum_yaw;";
      outfile << "vidHARRISglobalSum_x;vidHARRISglobalSum_y;vidHARRISglobalSum_z;vidHARRISglobalSum_roll;vidHARRISglobalSum_pitch;vidHARRISglobalSum_yaw;";
      

      outfile << "vidOFcorners2;vidOFmatches;vidOFinliers;";
      outfile << "vidSURFcorners2;vidSURFmatches;vidSURFinliers;";
      outfile << "vidSIFTcorners2;vidSIFTmatches;vidSIFTinliers;";
      outfile << "vidHARRIScorners2;vidHARRISmatches;vidHARRISinliers;";

      outfile << "OF_EstValid;SURF_EstValid;SIFT_EstValid;HARRIS_EstValid;";
      outfile << endl;

      //-------Open Out File for GTSAM, write column headings ------------------// 
      //------------------------------------------------------------------------//
      ofstream outfileCamGTSAM("outputCameraGTSAM.csv");
      outfileCamGTSAM << "length=" << ";" << numVidFrames-1 << ";" << endl; // First line should just be length                                                                       
      outfileCamGTSAM << "t1;t2;x;y;z;roll;pitch;yaw;numCorners;numMatches;numInliers;estValid;" << endl; // Second line is headings 

	//---------------------------------------------------------------------
	//----------------Create Video Writer---------------------------------
      VideoWriter outputVideoOF, outputVideoSURF;
      int outputFPS = 10;
      if(WRITE_VIDEO)
	{
	  outputVideoOF.open("outputOF.avi",cv::VideoWriter::fourcc('X','V','I','D'), outputFPS, Size(videoSize.width*2,videoSize.height*2),true);
	  if(!outputVideoOF.isOpened())
	    {
	      cout << "Could not open the output video for write" << endl;
	      return -1;
	    }

	  outputVideoSURF.open("outputSURF.avi",cv::VideoWriter::fourcc('X','V','I','D'), outputFPS, Size(videoSize.width*2,videoSize.height*2),true);
	  if(!outputVideoSURF.isOpened())
	    {
	      cout << "Could not open the output video for write" << endl;
	      return -1;
	    }
	}

	//---------------------------------------------------------------------
	//----------------Main Loop--------------------------------------------

	//Create Previous Image Containers:
	Mat prev_videoImgGray, videoImgGray; 
	int prev_i = 0;

	for (i = 0; i < numVidFrames; i++) {
          cout << i+1 << "/" << numVidFrames << endl;
	
	//-------------------------------------------
	//----------Get Video Frame i--------------//
	  //DEBUG:
	//cout << "CAP PROP POS FRAMES = " << inputVideo.get(CAP_PROP_POS_FRAMES) << endl;
	//cout << "CAP PROP MSEC" << inputVideo.get(CAP_PROP_POS_MSEC) << std::endl; //Print msec location in video
	  //END DEBUG

       	inputVideo.set(CAP_PROP_POS_FRAMES,i);
	inputVideo >> videoImg;
	
	//DEBUG:
	//	cout << "CAP PROP MSEC" << inputVideo.get(CAP_PROP_POS_MSEC) << std::endl; //Print msec location in video

	//FIX FOR BAD FRAMES AT END OF VIDEOS:
	while(videoImg.rows == 0) //Can also do .empty()
	  {
	    cout << "BAD FRAMES FOR WINDOWS DATA" << endl;
	    i++;
	    if(i >= numVidFrames)
		break;
	    inputVideo.set(CAP_PROP_POS_FRAMES, i);
	    inputVideo >> videoImg;
	  }
	if(i >= numVidFrames) //If at end, exit loop
	  {
	    cout << "Break from loop: i= " << i << endl;
	    break;
	  }

	//---------------------------------------------------------
	//-------------Preprocess Video Frame----------------------------
	//---------------------------------------------------------
	//Get gray image:
	cvtColor(videoImg, videoImgGray, COLOR_BGR2GRAY);

	if(WRITE_IMGS)
	  {
	    imwrite("VideoIMG.jpg", videoImg);
	    imwrite("VideoGrayIMG.jpg", videoImgGray);
	  }

	if(camDataType==12) //If Icefin Front Camera:
	  videoImgGray=videoImgGray.mul(icefinFrontMask);

	//DEBUG - Resize Image:
	//imshow(video_wnd, videoImgGray);
       	//resize(videoImgGray, videoImgGray, Size(720,480));
	//imshow(video_wnd, videoImgGray);
	//END DEBUG - Resize Image

	//Apply CLAHE:
	clahe->apply(videoImgGray,videoImgGray);
	if(WRITE_IMGS)
	  imwrite("VideoCLAHEIMG.jpg", videoImgGray);

	//Undistort the image:
	Mat videoImgUndistort;
	remap(videoImgGray, videoImgUndistort, map1, map2, INTER_LINEAR);
	videoImgGray = videoImgUndistort.clone();
	if(WRITE_IMGS)
	  imwrite("VideoUndistortIMG.jpg", videoImgGray);

	//Apply Blur:
	Size vidBlurKernelSize = Size(7,7);
       	if(BLUR_VID_IMGS && (camDataType != GTRI_LAPTOP) && (camDataType != BLENDER_DOWN) && (camDataType != BLENDER_FRONT)) //Don't blur laptop and blender imgs
	  GaussianBlur(videoImgGray,videoImgGray,vidBlurKernelSize,0,0); //Gaussian Blur
	if(WRITE_IMGS)
	  imwrite("VideoImgBlur.jpg", videoImgGray);

	//Fill in previous if first frame:
	if(i == 0)
	  {
	    cout << "I is 0 - saving fake previous image" << endl;
	    prev_videoImgGray = videoImgGray.clone();
	    prev_i = i;
	    if(!PROCESS_00_FRAME)
	      continue; //Don't process 0-0 frame. Go to 0-1 frame
	  }
	
	//----------------------------------------------------------
	//-----------------Get Video Shifts-------------------------
	//----------------------------------------------------------

	//------------------ Sparse Optical Flow Estimates ---------------//
	// Parameters for Shi-Tomasi algorithm                                      
        vector<Point2f> vidOFcorners1;
	vector<Point2f> vidOFcorners2;
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
	int vidOFmax_corners = 1000;
	Mat vidOFmodel, vidOFmask;
	double vidOFtheta_x1, vidOFtheta_y1, vidOFtheta_z1;
	double vidOFtheta_x2, vidOFtheta_y2, vidOFtheta_z2;
	Mat vidOFrotMat1, vidOFrotMat2;
	Mat vidOFtranslMat;
	double vidOFanglesArr[3] = {0,0,0};
	Mat rotatedVidOFangles = Mat::zeros(3,1,DataType<double>::type);
	Mat rotatedVidOFtransl = Mat::zeros(3,1,DataType<double>::type);
	int vidOFnumCorners = 0;
	int vidOFnumMatches = 0;
	int vidOFnumInliers = 0;

	//Corner Detection:
	goodFeaturesToTrack( prev_videoImgGray, vidOFcorners1, vidOFmax_corners, vidOFqualityLevel, vidOFminDistance, Mat(), vidOFblockSize, vidOFuseHarrisDetector, vidOF_k);

	vidOFnumCorners = vidOFcorners1.size();

        if(vidOFcorners1.size() > 0) //Detected some corners
	  {

	    //Calculate Corners Subpixel Accuracy:
	    Size vidOFsubPixWinSize(10,10);
	    cornerSubPix(prev_videoImgGray, vidOFcorners1, vidOFsubPixWinSize, Size(-1,-1), vidOFtermcrit);

	    //Lucas Kanade Pyramid Algorithm:        
	    vector<uchar> vidOFstatus;
	    vector<float> vidOFerr;

	    calcOpticalFlowPyrLK(prev_videoImgGray, videoImgGray, vidOFcorners1, vidOFcorners2, vidOFstatus, vidOFerr, vidOFwinSize, 7, vidOFtermcrit, 0, 0.001);

	    if(VERBOSE)
	      cout << "NUM VID OFlow CORNERS:" << vidOFnumCorners << endl;

	    std::vector< DMatch > vidOFgood_matches;
	    for(int i1=0; i1 < vidOFcorners1.size(); i1++)
	      {
		if(vidOFstatus[i1])
		  {
		    //cout << "corner1: " << vidOFcorners1[i1] << " -- corner2:" << vidOFcorners2[i1] << endl;
		    //Save Points into Keypoint Form For Drawing Matches:
		    vidOFkeypoints1.push_back(KeyPoint(vidOFcorners1[i1],1.f));
		    vidOFkeypoints2.push_back(KeyPoint(vidOFcorners2[i1],1.f));  
		    vidOFgoodMatches1.push_back(vidOFcorners1[i1]);
		    vidOFgoodMatches2.push_back(vidOFcorners2[i1]);
		    //    cout << vidOFcorners1[i1] << " --- " << vidOFcorners2[i1] << endl;
		  }
	      }
	    
	    vidOFnumMatches = vidOFgoodMatches1.size();

	    if(vidOFgoodMatches1.size() > 5) //If found some matches
	      {
		//Find Fundamental and Essential Matrices
		double vidOF_RANSAC_reprojthresh = 1;
		double vidOF_RANSAC_param = 0.99;

		//NOTE: findEssentialMatNew() is customized version of findEssentialMat() to enable change of fx, fy, princ_pt
		Mat vidOFessentialMatrix = findEssentialMatNew(vidOFgoodMatches1, vidOFgoodMatches2, fx_new, fy_new, princ_pt, ROBUST_EST_METH, vidOF_RANSAC_param, vidOF_RANSAC_reprojthresh,vidOFmask); 

		if(VERBOSE)
		  cout << "Video OF Essential Mat = " << vidOFessentialMatrix << endl;

		Mat tempMask = vidOFmask.clone();

		if(vidOFessentialMatrix.rows > 0) //If found an essential Mat
		  {
		    if(vidOFessentialMatrix.rows > 3)
		      {
			cout << "OF: " << vidOFessentialMatrix.rows/3 << " Essential Matrices Found!!!" << endl; //!!! FIND BETTER WAY OF HANDLING MULTIPLE IF NEEDED
			vidOFessentialMatrix.resize(3);
		      }
		    vidOFmodel = vidOFessentialMatrix.clone();
		    vidOFnumInliers = sum(vidOFmask)[0]; //Find number of inliers
		    if(VERBOSE)
		      cout << "vid OF num inliers = " << vidOFnumInliers << endl;

		    //Decompose the Essential Matrix into 2 possible Rotation Mats and a Translation Mat:
		    decomposeEssentialMat(vidOFessentialMatrix,vidOFrotMat1, vidOFrotMat2, vidOFtranslMat); //vidOFtranslMat);
		    
		    //NOTE: Tried to use recoverPose() instead of decomposeEssentialMat(), but didn't work. The translation is estimated the same, but recoverPose() is supposed to do chirality check to pick correct RotMat, but guesses wrong at least half the time. So I implemented my own check to find which of the RotMats to use instead. My check works well, but I should check it with real data more. Using recoverPose just for transl because it does Chirality check.
		    //NOTE: recoverPoseNew is customized version of recoverPose() to allow custom fx, fy instead of just focal.
		    Mat temp1, temp2;
		    recoverPoseNew(vidOFessentialMatrix, vidOFgoodMatches1, vidOFgoodMatches2, temp1, vidOFtranslMat, fx_new, fy_new, princ_pt, temp2);
		    OF_EstValid = 1;
		  }
		else 
		  {
		    cout << "No Vid OF Model Found" << endl;
		    vidOFessentialMatrix = Mat(3,3,DataType<double>::type,essentialMatZero);
		    vidOFmodel = Mat(3,3,DataType<double>::type,essentialMatZero);
		    vidOFrotMat1 = Mat(3,3,DataType<double>::type,rotZero);
		    vidOFrotMat2 = Mat(3,3,DataType<double>::type,rotZero);
		    vidOFtranslMat = Mat::zeros(3,1,CV_64F);
		    vidOFmask = Mat::zeros(1,vidOFgoodMatches1.size(),CV_8U);
		    vidOFnumInliers = 0;
		    OF_EstValid = 0;
		  }
		
		//Obtain the two sets of possible Euler angles from the Rot Mats:
		vidOFtheta_x1 = atan2(vidOFrotMat1.at<double>(2,1),vidOFrotMat1.at<double>(2,2))*180/CV_PI;
	  	vidOFtheta_y1 = atan2(-vidOFrotMat1.at<double>(2,0),sqrt(vidOFrotMat1.at<double>(2,1)*vidOFrotMat1.at<double>(2,1) + vidOFrotMat1.at<double>(2,2)*vidOFrotMat1.at<double>(2,2)))*180/CV_PI;
		vidOFtheta_z1 = atan2(vidOFrotMat1.at<double>(1,0),vidOFrotMat1.at<double>(0,0))*180/CV_PI;  

		vidOFtheta_x2 = atan2(vidOFrotMat2.at<double>(2,1),vidOFrotMat2.at<double>(2,2))*180/CV_PI;
	  	vidOFtheta_y2 = atan2(-vidOFrotMat2.at<double>(2,0),sqrt(vidOFrotMat2.at<double>(2,1)*vidOFrotMat2.at<double>(2,1) + vidOFrotMat2.at<double>(2,2)*vidOFrotMat2.at<double>(2,2)))*180/CV_PI;
		vidOFtheta_z2 = atan2(vidOFrotMat2.at<double>(1,0),vidOFrotMat2.at<double>(0,0))*180/CV_PI;  
		
		//Check to find the correct rotation - if magnitude of x and z rot is bigger, then it is the wrong matrix. Also check if any angles larger than 100 (bigger than FOV) which means wrong matrix: 
		//!!!!! NEEDS TO BE EVALUATED FUTHER ON REAL DATA. 
		int VALID_ROT_MAT_FLAG = 0; //Flag to choose between R1 and R2
		//If larger roll or pitch angles, less likely. These should be ~0.:
		if(FRONT_CAMERA) // Yaw is around y-axis (camera coordinates)
		  {
		    if((abs(vidOFtheta_x1) > abs(vidOFtheta_x2))&&(abs(vidOFtheta_z1) > abs(vidOFtheta_z2)))
		      VALID_ROT_MAT_FLAG = 2; 
		    else if((abs(vidOFtheta_x2) > abs(vidOFtheta_x1))&&(abs(vidOFtheta_z2) > abs(vidOFtheta_z1)))
		      VALID_ROT_MAT_FLAG = 1;
		    else if((vidOFtheta_x1 == 0) && (vidOFtheta_z1 == 0))
		      VALID_ROT_MAT_FLAG = 1;
		    else if((vidOFtheta_x2 == 0) && (vidOFtheta_z2 == 0))
		      VALID_ROT_MAT_FLAG = 2;
		  }
		else //Down Camera - Yaw is around z-axis (camera coordinates)
		  {
		    if((abs(vidOFtheta_x1) > abs(vidOFtheta_x2))&&(abs(vidOFtheta_y1) > abs(vidOFtheta_y2)))
		      VALID_ROT_MAT_FLAG = 2; 
		    else if((abs(vidOFtheta_x2) > abs(vidOFtheta_x1))&&(abs(vidOFtheta_y2) > abs(vidOFtheta_y1)))
		      VALID_ROT_MAT_FLAG = 1;
		    else if((vidOFtheta_x1 == 0) && (vidOFtheta_y1 == 0))
		      VALID_ROT_MAT_FLAG = 1;
		    else if((vidOFtheta_x2 == 0) && (vidOFtheta_y2 == 0))
		      VALID_ROT_MAT_FLAG = 2;
		  }

		//Frame to Frame angles > 100 very unlikely:
		if((abs(vidOFtheta_x1) > 100) || (abs(vidOFtheta_y1) > 100) || (abs(vidOFtheta_z1) > 100)) 
		  VALID_ROT_MAT_FLAG += 20; 
		if((abs(vidOFtheta_x2) > 100) || (abs(vidOFtheta_y2) > 100) || (abs(vidOFtheta_z2) > 100))
		  VALID_ROT_MAT_FLAG += 10;

		//Choose between the two possible rotation matrices based on flag:
		if((VALID_ROT_MAT_FLAG == 1) || (VALID_ROT_MAT_FLAG == 10) || (VALID_ROT_MAT_FLAG == 11))
		  {
		    if(VERBOSE)
		      cout << "Video OF Rotation Matrix: " << vidOFrotMat1 << endl;

		    vidOFanglesArr[0] = vidOFtheta_x1;
		    vidOFanglesArr[1] = vidOFtheta_y1;
		    vidOFanglesArr[2] = vidOFtheta_z1;
		  }
		else if((VALID_ROT_MAT_FLAG == 2) || (VALID_ROT_MAT_FLAG == 20) || (VALID_ROT_MAT_FLAG == 22))
		  { 
		    if(VERBOSE)
		      cout << "Video OF Rotation Matrix: " << vidOFrotMat2 << endl;

		    vidOFanglesArr[0] = vidOFtheta_x2;
		    vidOFanglesArr[1] = vidOFtheta_y2;
		    vidOFanglesArr[2] = vidOFtheta_z2;
		  }		   
		else 
		  { 
		    cout << "Video OF - NEITHER ROT MAT WORKS. Setting angles to 0,0,0. FLAG = " << VALID_ROT_MAT_FLAG << endl; 
		    cout << "Theta x1,y1,z1 = " << vidOFtheta_x1 << "," << vidOFtheta_y1 << "," << vidOFtheta_z1 << endl;
		    cout << "Theta x2,y2,z2 = " << vidOFtheta_x2 << "," << vidOFtheta_y2 << "," << vidOFtheta_z2 << endl;
		    vidOFanglesArr[0] = 0;
		    vidOFanglesArr[1] = 0;
		    vidOFanglesArr[2] = 0;
		    //exit(0);
		    OF_EstValid = 0;
		  }		   
		
		Mat vidOFangles = Mat(3,1,DataType<double>::type,vidOFanglesArr);
		if((camDataType == LINUX0DEG)||(camDataType == WINDOWS0DEG))
		  {
		    rotatedVidOFangles = RcamtoV*vidOFangles;
		    rotatedVidOFtransl = RcamtoV*vidOFtranslMat;
		  }
		if((camDataType == LINUX17DEG)||(camDataType == WINDOWS17DEG))
		  {
		    rotatedVidOFangles = RcamtoV*Rneg17x*vidOFangles;
		    rotatedVidOFtransl = RcamtoV*Rneg17x*vidOFtranslMat;
		  }
		if((camDataType == LINUX30DEG)||(camDataType == WINDOWS30DEG))
		  {
		    rotatedVidOFangles = RcamtoV*Rneg30x*vidOFangles;
		    rotatedVidOFtransl = RcamtoV*Rneg30x*vidOFtranslMat;
		  }
		if((camDataType == LINUX68DEG)||(camDataType == WINDOWS68DEG))
		  {
		    rotatedVidOFangles = RcamtoV*Rneg68x*vidOFangles;
		    rotatedVidOFtransl = RcamtoV*Rneg68x*vidOFtranslMat;
		  }
		if((camDataType == BLENDER_DOWN)||(camDataType == BLENDER_FRONT))
		  {
		    rotatedVidOFangles = RcamtoV*vidOFangles;
		    rotatedVidOFtransl = RcamtoV*vidOFtranslMat;
		  }
		if(camDataType == GTRI_LAPTOP)
		  {
		    rotatedVidOFangles = RcamtoV*Rneg20x*vidOFangles;
		    rotatedVidOFtransl = RcamtoV*Rneg20x*vidOFtranslMat;
		  }
		if((camDataType == GOPRO_BRITNEY)||(camDataType == ICEFIN_FRONT)||(camDataType == ICEFIN_DOWN))
		  {
		    rotatedVidOFangles = RcamtoV*vidOFangles;
		    rotatedVidOFtransl = RcamtoV*vidOFtranslMat;
		  }

		//Multiply R and t by -1 to get vehicle-relative instead of world-relative

		if(ZERO_ROLLPITCH)
		  {
		    rotatedVidOFangles.at<double>(0,0) = 0;
		    rotatedVidOFangles.at<double>(0,1) = 0;
		  }

		rotatedVidOFtransl = rotatedVidOFtransl*-1;
		rotatedVidOFangles = rotatedVidOFangles*-1;
		
		//Check for large jumps - Bad estimate
		if(THRESHOLD_JUMPS)
		  {
		    if((abs(rotatedVidOFangles.at<double>(0,0)) > MAX_ROT)||
		       (abs(rotatedVidOFangles.at<double>(0,1)) > MAX_ROT)||
		       (abs(rotatedVidOFangles.at<double>(0,2)) > MAX_ROT))
		      {
			cout << "OF Angles out of reasonable Range. Zeroed." << endl;
			cout << rotatedVidOFangles << endl;
			rotatedVidOFangles = Mat::zeros(3,1,CV_64F);
			OF_EstValid = 0;
		      }
		  }

		//Get the sum angles
		vidOFsumAngles = vidOFsumAngles + rotatedVidOFangles;

		if(VERBOSE)
		  {
		    cout << "OF VALID ROT MAT FLAG = " << VALID_ROT_MAT_FLAG << endl;
		    cout << "vidOFangles = " << vidOFangles << endl;
		    cout << "rotatedVidOFangles = " << rotatedVidOFangles << endl;
		    cout << "SumrotatedVidOFangles = " << vidOFsumAngles << endl;  
		    cout << "Video OF Translation: " << vidOFtranslMat << endl;
		    cout << "Video OF R1=" << vidOFrotMat1 << endl;
		    cout << "Video OF R2=" << vidOFrotMat2 << endl;
		    cout << "Video OF Theta_x1 and _x2: " << vidOFtheta_x1 << " -- " << vidOFtheta_x2 << endl;
		    cout << "Video OF Theta_y1 and _y2: " << vidOFtheta_y1 <<  " -- " << vidOFtheta_y2 << endl;
		    cout << "Video OF Theta_z1 and _z2: " << vidOFtheta_z1 <<  " -- " << vidOFtheta_z2 << endl;
		  }

		if(DISPLAY_IMGS)
		  {
		    //----------Show Features-----------//                            
		    cvtColor(prev_videoImgGray, vidFeaturesImg, CV_GRAY2BGR); //Get copy of gray img to mark features      
		    
		    //DEBUG: Use drawMatches instead of custom function
		    //drawMatches(prev_sonarImgGray, sonOFkeypoints1, sonarImgGray, sonOFkeypoints2, sonOFgood_matches, matchesImg,Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		    //END DEBUG: Use drawMatches
		    
		    //Draw Feature Matches (Can't use drawMatches() with OF):
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
			if(vidOFmask.at<char>(0,i1))
			  color = Scalar(0,255,0); //Green
			else
			  color = Scalar(0,0,255); //Red
			
			//Draw Features:
			circle(vidFeaturesImg, vidOFcorners1[i1], 4, color);
			
			//Draw Matches and lines between:
			circle(vidMatchesImg, vidOFgoodMatches1[i1], 4, color);
			circle(vidMatchesImg, vidOFgoodMatches2[i1] + vidOFimg2offset, 4, color);
			line(vidMatchesImg, vidOFgoodMatches1[i1], vidOFgoodMatches2[i1]+vidOFimg2offset, color);
		      }

		    vidMatchesImgOF = vidMatchesImg.clone();
		  }
	      }
	    else //If less than 5 matches found:
	      {
		cout << "NO OF MODEL FOUND" << endl;
		vidOFmodel = Mat(3,3,DataType<double>::type,essentialMatZero);
		vidOFtranslMat = Mat::zeros(3,1,CV_64F);
		OF_EstValid = 0;
	      }
	  }
	else //If no corners found:
	  {
	    cout << "NO OF MODEL FOUND" << endl;
	    vidOFmodel = Mat(3,3,DataType<double>::type,essentialMatZero);
	    vidOFtranslMat = Mat::zeros(3,1,CV_64F);
	    OF_EstValid = 0;
	  }
	
	if(VERBOSE)
	  {
	    //cout << "Vid OF Mask:" << endl << vidOFmask << endl;
	    cout << "Num Video OFlow Matches: " << vidOFgoodMatches1.size() << endl;
	    cout << "Num Video OFlow Inliers: " << vidOFnumInliers << endl;
	    cout << "Video OFlow Model: " << vidOFmodel << endl;
	  }

	if(DISPLAY_IMGS)
	  {
	    putText(vidFeaturesImg, "Optical Flow (Video)", Point(10,25),FONT_HERSHEY_SIMPLEX,1,Scalar(255,255,255));
	    putText(vidMatchesImg, "Optical Flow (Video)", Point(10,25),FONT_HERSHEY_SIMPLEX,1,Scalar(255,255,255));
	    imshow(vid_features_wnd, vidFeaturesImg);
	    imshow(vid_matches_wnd, vidMatchesImg);
	    if(STOP_BETWEEN_IMGS)
	      waitKey(0);
	    else
	      waitKey(10);
	  }
	if(WRITE_IMGS)
	  {
	    imwrite("OFvidFeaturesIMG.jpg", vidFeaturesImg);
	    imwrite("OFvidMatchesIMG.jpg", vidMatchesImg);
	  }
	
	//------------ SURF Feature Detection -----------------
	cvtColor(prev_videoImgGray, vidFeaturesImg, CV_GRAY2BGR); //Get copy of gray img to mark features      
	vidMatchesImg = prev_videoImgGray.clone();
	Mat vidSURFmodel, vidSURFmask;
	int vidSURFnumCorners = 0;
	int vidSURFnumMatches = 0;
	int vidSURFnumInliers = 0;
	double vidSURFtheta_x1, vidSURFtheta_y1, vidSURFtheta_z1;
	double vidSURFtheta_x2, vidSURFtheta_y2, vidSURFtheta_z2;
	Mat vidSURFrotMat1, vidSURFrotMat2;
	Mat vidSURFtranslMat;
	double vidSURFanglesArr[3] = {0,0,0};
	Mat rotatedVidSURFangles = Mat::zeros(3,1,DataType<double>::type);
	Mat rotatedVidSURFtransl = Mat::zeros(3,1,DataType<double>::type);

	std::vector<KeyPoint> vidSURFkeypoints_1, vidSURFkeypoints_2;
	vidSURFdetector.detect( prev_videoImgGray, vidSURFkeypoints_1 );
	vidSURFdetector.detect( videoImgGray, vidSURFkeypoints_2 );
	vidSURFnumCorners = vidSURFkeypoints_1.size();

	//SURF Calculate descriptors (feature vectors):                             
	Mat vidSURFdescriptors_1, vidSURFdescriptors_2;
	vidSURFextractor.compute( prev_videoImgGray, vidSURFkeypoints_1, vidSURFdescriptors_1 );
	vidSURFextractor.compute( videoImgGray, vidSURFkeypoints_2, vidSURFdescriptors_2 );
	
	if((!vidSURFdescriptors_1.empty()) && (!vidSURFdescriptors_2.empty()))
	  {
	    //SURF Matching descriptor vectors using FLANN matcher 
	    std::vector< DMatch > vidSURFmatches;
	    vidSURFmatcher.match( vidSURFdescriptors_1, vidSURFdescriptors_2, vidSURFmatches );

	    //DEBUG: Add in for radius match (only close matches)
	    /*
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
	      { if( vidSURFmatches[i1].distance <= max(2*vidSURFmin_dist, 0.02) )
		  { vidSURFgood_matches.push_back( vidSURFmatches[i1]); }
	      }   */
	    //END DEBUG: Add in for radius match (only close matches)

	    //Accept all matches as good matches:
	    std::vector< DMatch > vidSURFgood_matches;
	    for( int i1 = 0; i1 < vidSURFdescriptors_1.rows; i1++ )
	      {
		vidSURFgood_matches.push_back( vidSURFmatches[i1]);
	      }   

	    //NOTE: When first testing, developed estimateTranslationNew() in my_ptsetreg.cpp that estimates shifts given match_pts_diff. Only 2D shifts though. Not used here, needed 5DOF instead. Function is still there though. vidSURFok = estimateTranslationNew(vidSURFgood_match_pts_diff, vidSURFgood_match_pts_diff, vidSURFmodel, vidSURFmask, vidSURF_RANSAC_reprojthresh, vidSURF_RANSAC_param);

	    std::vector<Point2f> vidSURFgood_match_pts1;
	    std::vector<Point2f> vidSURFgood_match_pts2;
	    for( int i1 = 0; i1 < vidSURFgood_matches.size(); i1++ )
	      {
		//-- Get the keypoints from the good matches     
		vidSURFgood_match_pts1.push_back( vidSURFkeypoints_1[ vidSURFgood_matches[i1].queryIdx ].pt );
		vidSURFgood_match_pts2.push_back( vidSURFkeypoints_2[ vidSURFgood_matches[i1].trainIdx ].pt );
		//cout << vidSURFgood_match_pts1[i1] << " -- " << vidSURFgood_match_pts2[i1] << endl; //Print out all matching points
	      }
	    vidSURFnumMatches = vidSURFgood_match_pts1.size();
	        
	    //Find Fundamental and Essential Matrices
	    double vidSURF_RANSAC_reprojthresh = 1;
	    double vidSURF_RANSAC_param = 0.99;

	    if(vidSURFgood_matches.size() > 5) //If found enough matches...
	      {
		//NOTE: findEssentialMatNew is a custom version of findEssentialMat() which takes in a custom fx, fy, and princ_pt.
		Mat vidSURFessentialMatrix = findEssentialMatNew(vidSURFgood_match_pts1, vidSURFgood_match_pts2, fx_new, fy_new, princ_pt, ROBUST_EST_METH, vidSURF_RANSAC_param, vidSURF_RANSAC_reprojthresh,vidSURFmask);

		//cout << "SURF MASK" << vidSURFmask << endl;
		if(VERBOSE)
		  cout << "SURF Essential Mat = " << vidSURFessentialMatrix << endl;
		
		Mat tempMask = vidSURFmask.clone();

		if(vidSURFessentialMatrix.rows > 0) //If found an Essential Mat...
		  {
		    if(vidSURFessentialMatrix.rows > 3) //If found multiple E Mats:
		      {
			cout << "SURF: " << vidSURFessentialMatrix.rows/3 << " Essential Matrices Found!!!" << endl;
			vidSURFessentialMatrix.resize(3); //!!!Need to find way better way of dealing with multiple??
		      }
		    vidSURFmodel = vidSURFessentialMatrix.clone();
		    vidSURFnumInliers = sum(vidSURFmask)[0]; //Find number of inliers				
		    //Decompose the Essential Matrix into 2 possible Rotation Mats and a Translation Mat:   
		    decomposeEssentialMat(vidSURFessentialMatrix,vidSURFrotMat1, vidSURFrotMat2, vidSURFtranslMat);
		    //NOTE: Tried to use recoverPose() instead of decomposeEssentialMat(), but didn't work. The translation is estimated the same, but recoverPose() is supposed to do chirality check to pick correct RotMat, but guesses wrong at least half the time. So I implemented my own check to find which of the RotMats to use instead. My check works well, but I should check it with real data more.  Using recoverPose just for transl because it does Chirality check.
		    //NOTE: recoverPoseNew is customized version of recoverPose() to allow custom fx, fy instead of just focal.
		    Mat temp1, temp2;
		    recoverPoseNew(vidSURFessentialMatrix, vidSURFgood_match_pts1, vidSURFgood_match_pts2, temp1, vidSURFtranslMat, fx_new, fy_new, princ_pt, temp2);
		    SURF_EstValid = 1;
		  }
		else //Couldn't Find Essential Mat
		  {
		    vidSURFessentialMatrix = Mat(3,3,DataType<double>::type,essentialMatZero);
		    vidSURFmodel = Mat(3,3,DataType<double>::type,essentialMatZero);
		    vidSURFrotMat1 = Mat(3,3,DataType<double>::type,rotZero);
                    vidSURFrotMat2 = Mat(3,3,DataType<double>::type,rotZero);
		    vidSURFtranslMat = Mat::zeros(3,1,CV_64F);
		    vidSURFmask = Mat::zeros(1,vidSURFgood_match_pts2.size(),CV_8U);
		    vidSURFnumInliers = 0;
		    cout << "NO SURF FUNDAMENTAL MATRIX FOUND" << endl;
		    SURF_EstValid = 0;
		  }

		vidSURFtheta_x1 = atan2(vidSURFrotMat1.at<double>(2,1),vidSURFrotMat1.at<double>(2,2))*180/CV_PI;
		vidSURFtheta_y1 = atan2(-vidSURFrotMat1.at<double>(2,0),sqrt(vidSURFrotMat1.at<double>(2,1)*vidSURFrotMat1.at<double>(2,1) + vidSURFrotMat1.at<double>(2,2)*vidSURFrotMat1.at<double>(2,2)))*180/CV_PI;
		vidSURFtheta_z1 = atan2(vidSURFrotMat1.at<double>(1,0),vidSURFrotMat1.at<double>(0,0))*180/CV_PI;  
		
		vidSURFtheta_x2 = atan2(vidSURFrotMat2.at<double>(2,1),vidSURFrotMat2.at<double>(2,2))*180/CV_PI;
	  	vidSURFtheta_y2 = atan2(-vidSURFrotMat2.at<double>(2,0),sqrt(vidSURFrotMat2.at<double>(2,1)*vidSURFrotMat2.at<double>(2,1) + vidSURFrotMat2.at<double>(2,2)*vidSURFrotMat2.at<double>(2,2)))*180/CV_PI;
		vidSURFtheta_z2 = atan2(vidSURFrotMat2.at<double>(1,0),vidSURFrotMat2.at<double>(0,0))*180/CV_PI;  
		
		//Check to find the correct rotation - if magnitude of x and z rot is bigger, then it is the wrong matrix. Also check if any angles larger than 100 (bigger than FOV) which means wrong matrix. THIS IS NOT PROVEN
		//!!!!! NEEDS TO BE EVALUTATED FURTHER ON REAL DATA.
		int VALID_ROT_MAT_FLAG = 0; //Flag to choose between R1 and R2
		//If larger roll/pitch angles, less likely (should be ~0)
		if(FRONT_CAMERA) //Yaw is around y-axis (Cam Coordinates)
		  {
		    if((abs(vidSURFtheta_x1) > abs(vidSURFtheta_x2))&&(abs(vidSURFtheta_z1) > abs(vidSURFtheta_z2)))
		      VALID_ROT_MAT_FLAG = 2;
		    else if((abs(vidSURFtheta_x2) > abs(vidSURFtheta_x1))&&(abs(vidSURFtheta_z2) > abs(vidSURFtheta_z1)))
		      VALID_ROT_MAT_FLAG = 1;
		    else if((vidSURFtheta_x1 == 0) && (vidSURFtheta_z1 == 0))
		      VALID_ROT_MAT_FLAG = 1;
		    else if((vidSURFtheta_x2 == 0) && (vidSURFtheta_z2 == 0))
		      VALID_ROT_MAT_FLAG = 2;
		  }
		else // Down Camera - Yaw is around z-axis (cam Coordinates)
		  {
		    if((abs(vidSURFtheta_x1) > abs(vidSURFtheta_x2))&&(abs(vidSURFtheta_y1) > abs(vidSURFtheta_y2)))
		      VALID_ROT_MAT_FLAG = 2;
		    else if((abs(vidSURFtheta_x2) > abs(vidSURFtheta_x1))&&(abs(vidSURFtheta_y2) > abs(vidSURFtheta_y1)))
		      VALID_ROT_MAT_FLAG = 1;
		    else if((vidSURFtheta_x1 == 0) && (vidSURFtheta_y1 == 0))
		      VALID_ROT_MAT_FLAG = 1;
		    else if((vidSURFtheta_x2 == 0) && (vidSURFtheta_y2 == 0))
		      VALID_ROT_MAT_FLAG = 2;
		  }

		//Frame to frame angles > 100 very unlikely:
		if((abs(vidSURFtheta_x1) > 100) || (abs(vidSURFtheta_y1) > 100) || (abs(vidSURFtheta_z1) > 100))
		  VALID_ROT_MAT_FLAG += 20;
		if((abs(vidSURFtheta_x2) > 100) || (abs(vidSURFtheta_y2) > 100) || (abs(vidSURFtheta_z2) > 100))
		  VALID_ROT_MAT_FLAG += 10;

		//Choose between the two possible rotation matrices based on flag
		if((VALID_ROT_MAT_FLAG == 1) || (VALID_ROT_MAT_FLAG == 10) || (VALID_ROT_MAT_FLAG == 11))
		  { 
		    if(VERBOSE)
		      cout << "Video SURF Rotation Matrix: " << vidSURFrotMat1 << endl;
		    
		    vidSURFanglesArr[0] = vidSURFtheta_x1;
		    vidSURFanglesArr[1] = vidSURFtheta_y1;
		    vidSURFanglesArr[2] = vidSURFtheta_z1;
		  }
		else if((VALID_ROT_MAT_FLAG == 2) || (VALID_ROT_MAT_FLAG == 20) || (VALID_ROT_MAT_FLAG == 22))
		  { 
		    if(VERBOSE)
		      cout << "Video SURF Rotation Matrix: " << vidSURFrotMat2 << endl;
		    vidSURFanglesArr[0] = vidSURFtheta_x2;
		    vidSURFanglesArr[1] = vidSURFtheta_y2;
		    vidSURFanglesArr[2] = vidSURFtheta_z2;
		  }		   

		else
                  {
                    cout << "Video SURF - NEITHER ROT MAT WORKS. Setting angles to 0,0,0. FLAG = " << VALID_ROT_MAT_FLAG << endl;
                    cout << "Theta x1,y1,z1 = " << vidSURFtheta_x1 << "," << vidSURFtheta_y1 << "," << vidSURFtheta_z1 << endl;
                    cout << "Theta x2,y2,z2 = " << vidSURFtheta_x2 << "," << vidSURFtheta_y2 << "," << vidSURFtheta_z2 << endl;
                    vidSURFanglesArr[0] = 0;
                    vidSURFanglesArr[1] = 0;
                    vidSURFanglesArr[2] = 0;
		    SURF_EstValid = 0;
                    //exit(0);                   
                  }

		Mat vidSURFangles = Mat(3,1,DataType<double>::type,vidSURFanglesArr);
		if((camDataType == LINUX0DEG)||(camDataType == WINDOWS0DEG))
		  {
		    rotatedVidSURFangles = RcamtoV*vidSURFangles;
		    rotatedVidSURFtransl = RcamtoV*vidSURFtranslMat;
		  }
		if((camDataType == LINUX17DEG)||(camDataType == WINDOWS17DEG))
		  {
		    rotatedVidSURFangles = RcamtoV*Rneg17x*vidSURFangles;
		    rotatedVidSURFtransl = RcamtoV*Rneg17x*vidSURFtranslMat;
		  }
		if((camDataType == LINUX30DEG)||(camDataType == WINDOWS30DEG))
		  {
		    rotatedVidSURFangles = RcamtoV*Rneg30x*vidSURFangles;
		    rotatedVidSURFtransl = RcamtoV*Rneg30x*vidSURFtranslMat;
		  }
		if((camDataType == LINUX68DEG)||(camDataType == WINDOWS68DEG))
		  {
		    rotatedVidSURFangles = RcamtoV*Rneg68x*vidSURFangles;
		    rotatedVidSURFtransl = RcamtoV*Rneg68x*vidSURFtranslMat;
		  }
		if((camDataType == BLENDER_DOWN)||(camDataType == BLENDER_FRONT))
		  {
		    rotatedVidSURFangles = RcamtoV*vidSURFangles;
		    rotatedVidSURFtransl = RcamtoV*vidSURFtranslMat;
		  }
		if(camDataType == GTRI_LAPTOP)
		  {
		    rotatedVidSURFangles = RcamtoV*Rneg20x*vidSURFangles;
		    rotatedVidSURFtransl = RcamtoV*Rneg20x*vidSURFtranslMat;
		  }	   
		if((camDataType == GOPRO_BRITNEY)||(camDataType == ICEFIN_FRONT)||(camDataType == ICEFIN_DOWN))
		  {
		    rotatedVidSURFangles = RcamtoV*vidSURFangles;
		    rotatedVidSURFtransl = RcamtoV*vidSURFtranslMat;
		  }  

		//Multiply R and t by -1 to get vehicle-relative instead of world-relative

		if(ZERO_ROLLPITCH)
		  {
		    rotatedVidSURFangles.at<double>(0,0) = 0;
		    rotatedVidSURFangles.at<double>(0,1) = 0;
		  }

		rotatedVidSURFtransl = rotatedVidSURFtransl*-1;
		rotatedVidSURFangles = rotatedVidSURFangles*-1;

		//Check for large jumps - Bad estimate
		if(THRESHOLD_JUMPS)
		  {
		    if((abs(rotatedVidSURFangles.at<double>(0,0)) > MAX_ROT)||
		       (abs(rotatedVidSURFangles.at<double>(0,1)) > MAX_ROT)||
		       (abs(rotatedVidSURFangles.at<double>(0,2)) > MAX_ROT))
		      {
			cout << "SURF Angles out of reasonable Range. Zeroed." << endl;
			cout << rotatedVidSURFangles << endl;
			rotatedVidSURFangles = Mat::zeros(3,1,CV_64F);
			SURF_EstValid = 0;
		      }
		  }

		//Get the sum angles
		vidSURFsumAngles = vidSURFsumAngles + rotatedVidSURFangles;

		if(VERBOSE)
		  {
		    cout << "SURF VALID ROT MAT FLAG = " << VALID_ROT_MAT_FLAG << endl;
		    cout << "vidSURFangles = " << vidSURFangles << endl;
		    cout << "rotatedVidSURFangles = " << rotatedVidSURFangles << endl;
		    cout << "sumRotatedVidSURFangles = " << vidSURFsumAngles << endl;
		    cout << "Video SURF Translation: " << vidSURFtranslMat << endl;
		    cout << "Video SURF R1=" << vidSURFrotMat1 << endl;
		    cout << "Video SURF R2=" << vidSURFrotMat2 << endl;
		    cout << "Video SURF Theta_x1 and _x2: " << vidSURFtheta_x1 << " -- " << vidSURFtheta_x2 << endl;
		    cout << "Video SURF Theta_y1 and _x2: " << vidSURFtheta_y1 << " -- " << vidSURFtheta_y2 << endl;
		    cout << "Video SURF Theta_z1 and _x2: " << vidSURFtheta_z1 << " -- " << vidSURFtheta_z2 << endl;   
		  }

		if(DISPLAY_IMGS)
		  {
		    //---------Show Features---------------//

		    //Draw Matches: 
		    drawMatches(prev_videoImgGray, vidSURFkeypoints_1, videoImgGray, vidSURFkeypoints_2, vidSURFgood_matches, vidMatchesImg,Scalar::all(-1), Scalar::all(-1), vector<char>()); // , DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		    
		    //Draw Features:
		    for(int i1 = 0; i1 < vidSURFgood_match_pts2.size(); i1++)
		      {
			Scalar color;
			if(vidSURFmask.at<char>(0,i1))
			  color = Scalar(0,255,0); //Green
			else
			  color = Scalar(0,0,255); //Red
			circle(vidFeaturesImg, vidSURFgood_match_pts1[i1], 4, color);
		      }

		    vidMatchesImgSURF = vidMatchesImg.clone();
		  }
	      }
	    else // Found Less than 5 Matches 
	      {
		cout << "NO SURF MODEL FOUND" << endl;
		vidSURFmodel = Mat(3,3,DataType<double>::type,essentialMatZero);
		vidSURFtranslMat = Mat::zeros(3,1,CV_64F);
		SURF_EstValid = 0;
	      }
	  }
	else // Didn't Find any Features
	  {
	    cout << "NO SURF MODEL FOUND" << endl;
	    vidSURFmodel = Mat(3,3,DataType<double>::type,essentialMatZero);
	    vidSURFtranslMat = Mat::zeros(3,1,CV_64F);
	    SURF_EstValid = 0;
	  }
	
	if(VERBOSE)
	  {
	    //cout << "Vid SURF Mask: " << endl << vidSURFmask << endl;
	    cout << "Num SURF Corners: " << vidSURFnumCorners << endl;
	    cout << "Num SURF Matches: " << vidSURFnumMatches << endl;
	    cout << "Num SURF Inliers: " << vidSURFnumInliers << endl;
	    cout << "video SURF Model" << vidSURFmodel << endl;
	  }

	if(DISPLAY_IMGS)
	  {
	    putText(vidFeaturesImg, "SURF", Point(10,25),FONT_HERSHEY_SIMPLEX,1,Scalar(255,255,255));
	    putText(vidMatchesImg, "SURF", Point(10,25),FONT_HERSHEY_SIMPLEX,1,Scalar(255,255,255));
	   
	    imshow(vid_features_wnd, vidFeaturesImg);
	    imshow(vid_matches_wnd, vidMatchesImg);
	    if(STOP_BETWEEN_IMGS)
	      waitKey(0);
	    else
	      waitKey(10);
	  }
	if(WRITE_IMGS)
	  {
	    imwrite("SURFvidFeaturesIMG.jpg", vidFeaturesImg);
	    imwrite("SURFvidMatchesIMG.jpg", vidMatchesImg);
	  }
	
	//---------------video SIFT Feature Detection ------------------//
	cvtColor(prev_videoImgGray, vidFeaturesImg, CV_GRAY2BGR); //Get copy of gray img to mark features
	vidMatchesImg = prev_videoImgGray.clone();
        Mat vidSIFTmodel, vidSIFTmask;
	int vidSIFTnumCorners = 0;
	int vidSIFTnumInliers = 0;
	int vidSIFTnumMatches = 0;
	double vidSIFTtheta_x1, vidSIFTtheta_y1, vidSIFTtheta_z1;
	double vidSIFTtheta_x2, vidSIFTtheta_y2, vidSIFTtheta_z2;
	Mat vidSIFTrotMat1, vidSIFTrotMat2;
	Mat vidSIFTtranslMat;
	double vidSIFTanglesArr[3] = {0,0,0};
	Mat rotatedVidSIFTangles = Mat::zeros(3,1,DataType<double>::type);
	Mat rotatedVidSIFTtransl = Mat::zeros(3,1,DataType<double>::type);

	std::vector<KeyPoint> vidSIFTkeypoints_1, vidSIFTkeypoints_2;
        vidSIFTdetector->detect( prev_videoImgGray, vidSIFTkeypoints_1 );
        vidSIFTdetector->detect( videoImgGray, vidSIFTkeypoints_2 );

	vidSIFTnumCorners = vidSIFTkeypoints_1.size();

	//SIFT Calculate descriptors (feature vectors):
        Mat vidSIFTdescriptors_1, vidSIFTdescriptors_2;
        vidSIFTextractor->compute( prev_videoImgGray, vidSIFTkeypoints_1, vidSIFTdescriptors_1 );
        vidSIFTextractor->compute( videoImgGray, vidSIFTkeypoints_2, vidSIFTdescriptors_2 );

	if((!vidSIFTdescriptors_1.empty()) && (!vidSIFTdescriptors_2.empty()))
	  {
	    //SIFT Matching descriptor vectors using FLANN matcher
	    std::vector< DMatch > vidSIFTmatches;
	    vidSIFTmatcher.match( vidSIFTdescriptors_1, vidSIFTdescriptors_2, vidSIFTmatches );

	    //DEBUG: Add in for radius match (only close matches)
	    /*
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
	      { if( vidSIFTmatches[i1].distance <= max(2*vidSIFTmin_dist, 0.02) )
	          vidSIFTgood_matches.push_back( vidSIFTmatches[i1]);
	      }*/
	    //END DEBUG: Add in for radius match (only close matches)

	    //Accept all matches as good matches:
	    std::vector< DMatch > vidSIFTgood_matches;
	    for( int i1 = 0; i1 < vidSIFTdescriptors_1.rows; i1++ )
	      { 
		vidSIFTgood_matches.push_back( vidSIFTmatches[i1]);
	      }	      

	    //NOTE: When first testing, developed estimateTranslationNew() in my_ptsetreg.cpp that estimates shifts given match_pts_diff. Only 2D shifts though. Not used here, needed 5DOF instead. Function is still there though. vidSURFok = estimateTranslationNew(vidSURFgood_match_pts_diff, vidSURFgood_match_pts_diff, vidSURFmodel, vidSURFmask, vidSURF_RANSAC_reprojthresh, vidSURF_RANSAC_param); 

	    std::vector<Point2f> vidSIFTgood_match_pts1;
	    std::vector<Point2f> vidSIFTgood_match_pts2;
	    for( int i1 = 0; i1 < vidSIFTgood_matches.size(); i1++ )
	      {
		//-- Get the keypoints from the good matches
		vidSIFTgood_match_pts1.push_back( vidSIFTkeypoints_1[ vidSIFTgood_matches[i1].queryIdx ].pt );
		vidSIFTgood_match_pts2.push_back( vidSIFTkeypoints_2[ vidSIFTgood_matches[i1].trainIdx ].pt );
		//cout << vidSIFTgood_match_pts1[i1] << " -- " << vidSIFTgood_match_pts2[i1] << endl; //Print out all matching points
	      }
	    vidSIFTnumMatches = vidSIFTgood_match_pts1.size();
	    
	    //Find Fundamental and Essential Matrices:
	    double vidSIFT_RANSAC_reprojthresh = 1;
	    double vidSIFT_RANSAC_param = 0.99;
	    
	    if(vidSIFTgood_matches.size() > 5)
	      {
		//NOTE: findEssentialMatNew is a custom version of findEssentialMat() which takes in a custom fx, fy, and princ_pt.		
		Mat vidSIFTessentialMatrix = findEssentialMatNew(vidSIFTgood_match_pts1, vidSIFTgood_match_pts2, fx_new, fy_new, princ_pt, ROBUST_EST_METH, vidSIFT_RANSAC_param, vidSIFT_RANSAC_reprojthresh,vidSIFTmask); //ADD MASK LATER

		if(VERBOSE)
		  cout << "SIFT Essential Mat = " << vidSIFTessentialMatrix << endl;

		Mat tempMask = vidSIFTmask.clone();
		
		if(vidSIFTessentialMatrix.rows > 0) //If found an Essential Mat...
		  {
		    if(vidSIFTessentialMatrix.rows > 3) //If found multiple E Mats:
		      {	
			cout << "SIFT: " << vidSIFTessentialMatrix.rows/3 << "Essential Matrices Found!!!" << endl;
			vidSIFTessentialMatrix.resize(3); //!!!!Need to find way better way of dealing with multiple??
		      }
		    vidSIFTmodel = vidSIFTessentialMatrix.clone();
		    vidSIFTnumInliers = sum(vidSIFTmask)[0]; //Find number of inliers
		    //Decompose the Essential Matrix into 2 possible Rot Mats and a Translation Mat
		    decomposeEssentialMat(vidSIFTessentialMatrix, vidSIFTrotMat1, vidSIFTrotMat2, vidSIFTtranslMat);

		    //NOTE: Tried to use recoverPose() instead of decomposeEssentialMat(), but didn't work. The translation is estimated the same, but recoverPose() is supposed to do chirality check to pick correct RotMat, but guesses wrong at least half the time. So I implemented my own check to find which of the RotMats to use instead. My check works well, but I should check it with real data more. Use recoverPose just for transl to do chirality check.                
		    //NOTE: recoverPoseNew is customized version of recoverPose() to allow custom fx, fy instead of just focal.
		    Mat temp1, temp2; 
		    recoverPoseNew(vidSIFTessentialMatrix, vidSIFTgood_match_pts1, vidSIFTgood_match_pts2, temp1, vidSIFTtranslMat, fx_new, fy_new, princ_pt, temp2);
		    SIFT_EstValid = 1;
		  }
		else //Couldn't find Essential Mat
		  {
		    vidSIFTessentialMatrix = Mat(3,3,DataType<double>::type,essentialMatZero);
		    vidSIFTmodel = Mat(3,3,DataType<double>::type,essentialMatZero);
		    vidSIFTrotMat1 = Mat(3,3,DataType<double>::type,rotZero);
		    vidSIFTrotMat2 = Mat(3,3,DataType<double>::type,rotZero);
		    vidSIFTtranslMat = Mat::zeros(3,1,CV_64F);
		    vidSIFTmask = Mat::zeros(1,vidSIFTgood_match_pts2.size(),CV_8U);
		    vidSIFTnumInliers = 0;
		    cout << "NO SIFT FUNDAMENTAL MATRIX FOUND" << endl;
		    cout << "E set to:" << endl << vidSIFTmodel << endl;
		    SIFT_EstValid = 0;
		  }
		
		vidSIFTtheta_x1 = atan2(vidSIFTrotMat1.at<double>(2,1),vidSIFTrotMat1.at<double>(2,2))*180/CV_PI;
		vidSIFTtheta_y1 = atan2(-vidSIFTrotMat1.at<double>(2,0),sqrt(vidSIFTrotMat1.at<double>(2,1)*vidSIFTrotMat1.at<double>(2,1) + vidSIFTrotMat1.at<double>(2,2)*vidSIFTrotMat1.at<double>(2,2)))*180/CV_PI;
		vidSIFTtheta_z1 = atan2(vidSIFTrotMat1.at<double>(1,0),vidSIFTrotMat1.at<double>(0,0))*180/CV_PI;
		
		vidSIFTtheta_x2 = atan2(vidSIFTrotMat2.at<double>(2,1),vidSIFTrotMat2.at<double>(2,2))*180/CV_PI;
		vidSIFTtheta_y2 = atan2(-vidSIFTrotMat2.at<double>(2,0),sqrt(vidSIFTrotMat2.at<double>(2,1)*vidSIFTrotMat2.at<double>(2,1) + vidSIFTrotMat2.at<double>(2,2)*vidSIFTrotMat2.at<double>(2,2)))*180/CV_PI;
		vidSIFTtheta_z2 = atan2(vidSIFTrotMat2.at<double>(1,0),vidSIFTrotMat2.at<double>(0,0))*180/CV_PI;
		
		//Check to find the correct rotation - if magnitude of x and z rot is bigger, then it is the wrong matrix. Also check if any angles larger than 100 (bigger than FOV) which means wrong matrix.
		//!!!!!!! NEEDS TO BE EVALUATED FURTHER ON REAL DATA. 
		int VALID_ROT_MAT_FLAG = 0; //Flag to choose between R1 and R2
		//If larger roll/pitch angles, less likely (should be ~0)
		if(FRONT_CAMERA) //Yaw around y-axis
		  {
		    if((abs(vidSIFTtheta_x1) > abs(vidSIFTtheta_x2))&&(abs(vidSIFTtheta_z1) > abs(vidSIFTtheta_z2)))
		      VALID_ROT_MAT_FLAG = 2;
		    else if((abs(vidSIFTtheta_x2) > abs(vidSIFTtheta_x1))&&(abs(vidSIFTtheta_z2) > abs(vidSIFTtheta_z1)))
		      VALID_ROT_MAT_FLAG = 1;
		    else if((vidSIFTtheta_x1 == 0) && (vidSIFTtheta_z1 == 0))
		      VALID_ROT_MAT_FLAG = 1;
		    else if((vidSIFTtheta_x2 == 0) && (vidSIFTtheta_z2 == 0))
		      VALID_ROT_MAT_FLAG = 2;
		  }
		else //Down Camera - Yaw around z-axis (cam Coordinates)
		  {
		    if((abs(vidSIFTtheta_x1) > abs(vidSIFTtheta_x2))&&(abs(vidSIFTtheta_y1) > abs(vidSIFTtheta_y2)))
		      VALID_ROT_MAT_FLAG = 2;
		    else if((abs(vidSIFTtheta_x2) > abs(vidSIFTtheta_x1))&&(abs(vidSIFTtheta_y2) > abs(vidSIFTtheta_y1)))
		      VALID_ROT_MAT_FLAG = 1;
		    else if((vidSIFTtheta_x1 == 0) && (vidSIFTtheta_y1 == 0))
		      VALID_ROT_MAT_FLAG = 1;
		    else if((vidSIFTtheta_x2 == 0) && (vidSIFTtheta_y2 == 0))
		      VALID_ROT_MAT_FLAG = 2;
		  }

		//Frame to frame angles > 100 very unlikely
		if((abs(vidSIFTtheta_x1) > 100) || (abs(vidSIFTtheta_y1) > 100) || (abs(vidSIFTtheta_z1) > 100))
		  VALID_ROT_MAT_FLAG += 20;
		if((abs(vidSIFTtheta_x2) > 100) || (abs(vidSIFTtheta_y2) > 100) || (abs(vidSIFTtheta_z2) > 100))
		  VALID_ROT_MAT_FLAG += 10;
		
		//Choose between the two possible rotation matrices based on flag
		if((VALID_ROT_MAT_FLAG == 1) || (VALID_ROT_MAT_FLAG == 10) || (VALID_ROT_MAT_FLAG == 11))
		  {
		    if(VERBOSE)
		      cout << "Video SIFT Rotation Matrix: " << vidSIFTrotMat1 << endl;

		    vidSIFTanglesArr[0] = vidSIFTtheta_x1;
		    vidSIFTanglesArr[1] = vidSIFTtheta_y1;
		    vidSIFTanglesArr[2] = vidSIFTtheta_z1;
		  }
		else if((VALID_ROT_MAT_FLAG == 2) || (VALID_ROT_MAT_FLAG == 20) || (VALID_ROT_MAT_FLAG == 22))
		  {
		    if(VERBOSE)
		      cout << "Video SIFT Rotation Matrix: " << vidSIFTrotMat2 << endl;
		    vidSIFTanglesArr[0] = vidSIFTtheta_x2;
		    vidSIFTanglesArr[1] = vidSIFTtheta_y2;
		    vidSIFTanglesArr[2] = vidSIFTtheta_z2;
		  }	
		else
                  {
                    cout << "Video SIFT - NEITHER ROT MAT WORKS. Setting angles to 0,0,0. FLAG = " << VALID_ROT_MAT_FLAG << endl;
                    cout << "Theta x1,y1,z1 = " << vidSIFTtheta_x1 << "," << vidSIFTtheta_y1 << "," << vidSIFTtheta_z1 << endl;
                    cout << "Theta x2,y2,z2 = " << vidSIFTtheta_x2 << "," << vidSIFTtheta_y2 << "," << vidSIFTtheta_z2 << endl;
                    vidSIFTanglesArr[0] = 0;
                    vidSIFTanglesArr[1] = 0;
                    vidSIFTanglesArr[2] = 0;
		    SIFT_EstValid = 0;
                    //exit(0);                 
                  }
		
		Mat vidSIFTangles = Mat(3,1,DataType<double>::type,vidSIFTanglesArr);
		if((camDataType == LINUX0DEG)||(camDataType == WINDOWS0DEG))
		  {
		    rotatedVidSIFTangles = RcamtoV*vidSIFTangles;
		    rotatedVidSIFTtransl = RcamtoV*vidSIFTtranslMat;
		  }
		if((camDataType == LINUX17DEG)||(camDataType == WINDOWS17DEG))
		  {
		    rotatedVidSIFTangles = RcamtoV*Rneg17x*vidSIFTangles;
		    rotatedVidSIFTtransl = RcamtoV*Rneg17x*vidSIFTtranslMat;
		  }
		if((camDataType == LINUX30DEG)||(camDataType == WINDOWS30DEG))
		  {
		    rotatedVidSIFTangles = RcamtoV*Rneg30x*vidSIFTangles;
		    rotatedVidSIFTtransl = RcamtoV*Rneg30x*vidSIFTtranslMat;
		  }
		if((camDataType == LINUX68DEG)||(camDataType == WINDOWS68DEG))
		  {
		    rotatedVidSIFTangles = RcamtoV*Rneg68x*vidSIFTangles;
		    rotatedVidSIFTtransl = RcamtoV*Rneg68x*vidSIFTtranslMat;
		  }
		if((camDataType == BLENDER_DOWN)||(camDataType == BLENDER_FRONT))
		  {
		    rotatedVidSIFTangles = RcamtoV*vidSIFTangles;
		    rotatedVidSIFTtransl = RcamtoV*vidSIFTtranslMat;
		  }
		if(camDataType == GTRI_LAPTOP)
		  {
		    rotatedVidSIFTangles = RcamtoV*Rneg20x*vidSIFTangles;
		    rotatedVidSIFTtransl = RcamtoV*Rneg20x*vidSIFTtranslMat;
		  }	   
		if((camDataType == GOPRO_BRITNEY)||(camDataType == ICEFIN_FRONT)||(camDataType == ICEFIN_DOWN))
		  {
		    rotatedVidSIFTangles = RcamtoV*vidSIFTangles;
		    rotatedVidSIFTtransl = RcamtoV*vidSIFTtranslMat;
		  }  

		//Multiply R and t by -1 to get vehicle-relative instead of world-relative

		if(ZERO_ROLLPITCH)
		  {
		    rotatedVidSIFTangles.at<double>(0,0) = 0;
		    rotatedVidSIFTangles.at<double>(0,1) = 0;
		  }

		rotatedVidSIFTtransl = rotatedVidSIFTtransl*-1;
		rotatedVidSIFTangles = rotatedVidSIFTangles*-1;

		//Check for large jumps - Bad estimate
		if(THRESHOLD_JUMPS)
		  {
		    if((abs(rotatedVidSIFTangles.at<double>(0,0)) > MAX_ROT)||
		       (abs(rotatedVidSIFTangles.at<double>(0,1)) > MAX_ROT)||
		       (abs(rotatedVidSIFTangles.at<double>(0,2)) > MAX_ROT))
		      {
			cout << "SIFT Angles out of reasonable Range. Zeroed." << endl;
			cout << rotatedVidSIFTangles << endl;
			rotatedVidSIFTangles = Mat::zeros(3,1,CV_64F);
			SIFT_EstValid = 0;
		      }
		  }

		//Get the sum angles
		vidSIFTsumAngles = vidSIFTsumAngles + rotatedVidSIFTangles;
	
		if(VERBOSE)
		  {
		    cout << "SIFT VALID ROT MAT FLAG = " << VALID_ROT_MAT_FLAG << endl;
		    cout << "vidSIFTangles = " << vidSIFTangles << endl;
		    cout << "rotatedVidSIFTangles = " << rotatedVidSIFTangles << endl;
		    cout << "sumRotatedVidSIFTangles = " << vidSIFTsumAngles << endl;
		    cout << "Video SIFT Translation: " << vidSIFTtranslMat << endl;
		    cout << "Video SIFT R1=" << vidSIFTrotMat1 << endl;
		    cout << "Video SIFT R2=" << vidSIFTrotMat2 << endl;
		    cout << "Video SIFT Theta_x1 and _x2: " << vidSIFTtheta_x1 << " -- " << vidSIFTtheta_x2 << endl;
		    cout << "Video SIFT Theta_y1 and _y2: " << vidSIFTtheta_y1 << " -- " << vidSIFTtheta_y2 << endl;
		    cout << "Video SIFT Theta_z1 and _z2: " << vidSIFTtheta_z1 << " -- " << vidSIFTtheta_z2<< endl;	    
		  }
		
		if(DISPLAY_IMGS)
		  {
		    //------------Show Features-------------//

		    //Draw Matches:
		    drawMatches(prev_videoImgGray, vidSIFTkeypoints_1, videoImgGray, vidSIFTkeypoints_2, vidSIFTgood_matches, vidMatchesImg,Scalar::all(-1), Scalar::all(-1), vector<char>()); //, DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

		    //Draw Features:
		    for(int i1 = 0; i1 < vidSIFTgood_match_pts1.size(); i1++)
		      {
			Scalar color;
			if(vidSIFTmask.at<char>(0,i1))
			  color = Scalar(0,255,0); // Green
			else
			  color = Scalar(0,0,255); // Red
			circle(vidFeaturesImg, vidSIFTgood_match_pts1[i1], 4, color);
		      }
		  }
	      }
	    else //Found less than 5 matches
	      {
		cout << "NO SIFT MODEL FOUND" << endl;
		vidSIFTmodel = Mat(3,3,DataType<double>::type,essentialMatZero);
		cout << "model=" << endl << vidSIFTmodel << endl;
		vidSIFTtranslMat = Mat::zeros(3,1,CV_64F);
		SIFT_EstValid = 0;
	      }
	  }
	else // Didn't find any features
	  {
	    cout << "NO SIFT MODEL FOUND" << endl;
	    vidSIFTmodel = Mat(3,3,DataType<double>::type,essentialMatZero);
	    vidSIFTtranslMat = Mat::zeros(3,1,CV_64F);
	    SIFT_EstValid = 0;
	  }

	if(VERBOSE)
	  {
	    //cout << "Vid SIFT Mask: " << endl << vidSIFTmask << endl;
	    cout << "Num SIFT Matches: " << vidSIFTnumMatches << endl;
	    cout << "Num SIFT Inliers: " << vidSIFTnumInliers << endl;
	    cout << "video SIFT model" << vidSIFTmodel << endl;
	  }

	if(DISPLAY_IMGS)
	  {
	    putText(vidFeaturesImg, "SIFT", Point(10,25),FONT_HERSHEY_SIMPLEX,1,Scalar(255,255,255));
	    putText(vidMatchesImg, "SIFT", Point(10,25),FONT_HERSHEY_SIMPLEX,1,Scalar(255,255,255));

	    imshow(vid_features_wnd, vidFeaturesImg);
	    imshow(vid_matches_wnd, vidMatchesImg);
	    if(STOP_BETWEEN_IMGS)
	      waitKey(0);
	    else
	      waitKey(10);
	  }
	if(WRITE_IMGS)
	  {
	    imwrite("SIFTvidFeaturesIMG.jpg", vidFeaturesImg);
	    imwrite("SIFTvidMatchesIMG.jpg", vidMatchesImg);
	  }

	//-------------------------- HARRIS/SIFT Detection ---------------//
	//-- create detector and descriptor --
	Mat vidHARRISmodel, vidHARRISmask;
	int vidHARRISnumCorners = 0;
	int vidHARRISnumMatches = 0;
	int vidHARRISnumInliers = 0;
	double vidHARRIStheta_x1, vidHARRIStheta_y1, vidHARRIStheta_z1;
	double vidHARRIStheta_x2, vidHARRIStheta_y2, vidHARRIStheta_z2;
	Mat vidHARRISrotMat1, vidHARRISrotMat2;
	Mat vidHARRIStranslMat;
	double vidHARRISanglesArr[3] = {0,0,0};
	Mat rotatedVidHARRISangles = Mat::zeros(3,1,DataType<double>::type);
	Mat rotatedVidHARRIStransl = Mat::zeros(3,1,DataType<double>::type);

	cvtColor(prev_videoImgGray, vidFeaturesImg, CV_GRAY2BGR); //Get copy of gray img to mark features
	vidMatchesImg = prev_videoImgGray.clone();

	// detect keypoints
	std::vector<cv::KeyPoint> vidHARRISkeypoints1, vidHARRISkeypoints2;
	vidHARRISdetector->detect(prev_videoImgGray, vidHARRISkeypoints1);
	vidHARRISdetector->detect(videoImgGray, vidHARRISkeypoints2);

	vidHARRISnumCorners = vidHARRISkeypoints1.size();

	// extract feature descriptions
	cv::Mat vidHARRISdesc1, vidHARRISdesc2;
	vidHARRISdescriptor->compute(prev_videoImgGray, vidHARRISkeypoints1, vidHARRISdesc1);
	vidHARRISdescriptor->compute(videoImgGray, vidHARRISkeypoints2, vidHARRISdesc2);

	if((!vidHARRISdesc1.empty()) && (!vidHARRISdesc2.empty()))
	  {
	    //HARRIS Matching descriptor vectors using FLANN matcher 
	    std::vector< DMatch > vidHARRISmatches;
	    vidHARRISmatcher.match( vidHARRISdesc1, vidHARRISdesc2, vidHARRISmatches );

	    //DEBUG: Add in for radius match (only close matches)
	    /*
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
	      { 
	       if( vidHARRISmatches[i1].distance <= max(2*vidHARRISmin_dist, 0.02) )
	        { vidHARRISgood_matches.push_back( vidHARRISmatches[i1]); }
	      }
	    */
	    //END DEBUG: Add in for radius match (only close matches)

	    //Accept all matches as good matches
	    std::vector< DMatch > vidHARRISgood_matches;
	    for( int i1 = 0; i1 < vidHARRISdesc1.rows; i1++ )
	      { 
		vidHARRISgood_matches.push_back( vidHARRISmatches[i1]);
	      }

	    //NOTE: When first testing, developed estimateTranslationNew() in my_ptsetreg.cpp that estimates shifts given match_pts_diff. Only 2D shifts though. Not used here, needed 5DOF instead. Function is still there though. vidSURFok = estimateTranslationNew(vidSURFgood_match_pts_diff, vidSURFgood_match_pts_diff, vidSURFmodel, vidSURFmask, vidSURF_RANSAC_reprojthresh, vidSURF_RANSAC_param);   

	    std::vector<Point2f> vidHARRISgood_match_pts1;
	    std::vector<Point2f> vidHARRISgood_match_pts2;	        
	    for( int i1 = 0; i1 < vidHARRISgood_matches.size(); i1++ )
	      {
		//-- Get the keypoints from the good matches
		vidHARRISgood_match_pts1.push_back( vidHARRISkeypoints1[ vidHARRISgood_matches[i1].queryIdx ].pt );
		vidHARRISgood_match_pts2.push_back( vidHARRISkeypoints2[ vidHARRISgood_matches[i1].trainIdx ].pt );
	      }
	    vidHARRISnumMatches = vidHARRISgood_match_pts1.size();

	    //Find Fundamental and Essential Matrices                    
	    double vidHARRIS_RANSAC_reprojthresh = 1;
	    double vidHARRIS_RANSAC_param = 0.99;
	   
	    if(vidHARRISgood_matches.size() > 5) //If found enough matches...
	      {
		//NOTE: findEssentialMatNew is a custom version of findEssentialMat() which takes in a custom fx, fy and princ_pt
		Mat vidHARRISessentialMatrix = findEssentialMatNew(vidHARRISgood_match_pts1, vidHARRISgood_match_pts2, fx_new, fy_new, princ_pt, ROBUST_EST_METH, vidHARRIS_RANSAC_param, vidHARRIS_RANSAC_reprojthresh,vidHARRISmask); 
		
		if(VERBOSE)
		  cout << "HARRIS Essential Mat = " << vidHARRISessentialMatrix << endl;
		Mat tempMask = vidHARRISmask.clone();
		
		if(vidHARRISessentialMatrix.rows > 0) //If found an Essential Mat
		  {
		    if(vidHARRISessentialMatrix.rows > 3) //If found multiple E Mats
		      {
			cout << "HARRIS: " << vidHARRISessentialMatrix.rows/3 << " Essential Matrices Found!!!" << endl;
			vidHARRISessentialMatrix.resize(3); //!!!!Need to find a better way of dealing with multiple
		      }
		    vidHARRISmodel = vidHARRISessentialMatrix.clone();
		    vidHARRISnumInliers = sum(vidHARRISmask)[0]; //Find number of inliers		
		    //Decompose the Essential Matrix into 2 possible Rotation Mats and a Translation Mat
		    decomposeEssentialMat(vidHARRISessentialMatrix, vidHARRISrotMat1, vidHARRISrotMat2, vidHARRIStranslMat);

		    //NOTE: Tried to use recoverPose() instead of decomposeEssentialMat(), but didn't work. The translation is estimated the same, but recoverPose() is supposed to do chirality check to pick correct RotMat, but guesses wrong at least half the time. So I implemented my own check to find which of the RotMats to use instead. My check works well, but I should check it with real data more.  Use recoverPose just for transl to do chirality check.                
		    //NOTE: recoverPoseNew is customized version of recoverPose() to allow custom fx, fy instead of just focal. 
		    Mat temp1, temp2;
		    recoverPoseNew(vidHARRISessentialMatrix, vidHARRISgood_match_pts1, vidHARRISgood_match_pts2, temp1, vidHARRIStranslMat, fx_new, fy_new, princ_pt, temp2);
		    HARRIS_EstValid = 1;
		  }
		else //Couldn't find essential mat
		  {
		    vidHARRISessentialMatrix = Mat(3,3,DataType<double>::type,essentialMatZero);
		    vidHARRISmodel = Mat(3,3,DataType<double>::type,essentialMatZero);
		    vidHARRISrotMat1 = Mat(3,3,DataType<double>::type,rotZero);
                    vidHARRISrotMat2 = Mat(3,3,DataType<double>::type,rotZero);
		    vidHARRIStranslMat = Mat::zeros(3,1,CV_64F);
		    vidHARRISmask = Mat::zeros(1,vidHARRISgood_match_pts2.size(),CV_8U);
		    vidHARRISnumInliers = 0;
		    cout << "NO HARRIS FUNDAMENTAL MATRIX FOUND" << endl;
		    HARRIS_EstValid = 0;
		  }

		vidHARRIStheta_x1 = atan2(vidHARRISrotMat1.at<double>(2,1),vidHARRISrotMat1.at<double>(2,2))*180/CV_PI;
		vidHARRIStheta_y1 = atan2(-vidHARRISrotMat1.at<double>(2,0),sqrt(vidHARRISrotMat1.at<double>(2,1)*vidHARRISrotMat1.at<double>(2,1) + vidHARRISrotMat1.at<double>(2,2)*vidHARRISrotMat1.at<double>(2,2)))*180/CV_PI;
		vidHARRIStheta_z1 = atan2(vidHARRISrotMat1.at<double>(1,0),vidHARRISrotMat1.at<double>(0,0))*180/CV_PI;  
		
		vidHARRIStheta_x2 = atan2(vidHARRISrotMat2.at<double>(2,1),vidHARRISrotMat2.at<double>(2,2))*180/CV_PI;
	  	vidHARRIStheta_y2 = atan2(-vidHARRISrotMat2.at<double>(2,0),sqrt(vidHARRISrotMat2.at<double>(2,1)*vidHARRISrotMat2.at<double>(2,1) + vidHARRISrotMat2.at<double>(2,2)*vidHARRISrotMat2.at<double>(2,2)))*180/CV_PI;
		vidHARRIStheta_z2 = atan2(vidHARRISrotMat2.at<double>(1,0),vidHARRISrotMat2.at<double>(0,0))*180/CV_PI;  
		
		//Check to find the correct rotation - if magnitude of x and z rot is bigger, then it is the wrong matrix. Also check if any angles larger than 100 (bigger than FOV) which means wrong matrix. THIS IS NOT PROVEN
		//!!!!! NEEDS TO BE EVALUATED FURTHER ON REAL DATA. 
		int VALID_ROT_MAT_FLAG = 0; //Flag to choose between R1 and R2
		//If larger roll/pitch angles, less likely (should be ~0)
		if(FRONT_CAMERA) // Yaw around y-axis
		  {
		    if((abs(vidHARRIStheta_x1) > abs(vidHARRIStheta_x2))&&(abs(vidHARRIStheta_z1) > abs(vidHARRIStheta_z2)))
		      VALID_ROT_MAT_FLAG = 2;
		    else if((abs(vidHARRIStheta_x2) > abs(vidHARRIStheta_x1))&&(abs(vidHARRIStheta_z2) > abs(vidHARRIStheta_z1)))
		      VALID_ROT_MAT_FLAG = 1;
		    else if((vidHARRIStheta_x1 == 0) && (vidHARRIStheta_z1 == 0))
		      VALID_ROT_MAT_FLAG = 1;
		    else if((vidHARRIStheta_x2 == 0) && (vidHARRIStheta_z2 == 0))
		      VALID_ROT_MAT_FLAG = 2;
		  }
		else // DOWN CAM - Yaw around z-axis (cam Coordinates)
		  {
		    if((abs(vidHARRIStheta_x1) > abs(vidHARRIStheta_x2))&&(abs(vidHARRIStheta_y1) > abs(vidHARRIStheta_y2)))
		      VALID_ROT_MAT_FLAG = 2;
		    else if((abs(vidHARRIStheta_x2) > abs(vidHARRIStheta_x1))&&(abs(vidHARRIStheta_y2) > abs(vidHARRIStheta_y1)))
		      VALID_ROT_MAT_FLAG = 1;
		    else if((vidHARRIStheta_x1 == 0) && (vidHARRIStheta_y1 == 0))
		      VALID_ROT_MAT_FLAG = 1;
		    else if((vidHARRIStheta_x2 == 0) && (vidHARRIStheta_y2 == 0))
		      VALID_ROT_MAT_FLAG = 2;
		  }

		//Frame to frame angles > 100 very unlikely:
		if((abs(vidHARRIStheta_x1) > 100) || (abs(vidHARRIStheta_y1) > 100) || (abs(vidHARRIStheta_z1) > 100))
		  VALID_ROT_MAT_FLAG += 20;
		if((abs(vidHARRIStheta_x2) > 100) || (abs(vidHARRIStheta_y2) > 100) || (abs(vidHARRIStheta_z2) > 100))
		  VALID_ROT_MAT_FLAG += 10;

		//Choose between the two possible rotation matrices based on flag
		if((VALID_ROT_MAT_FLAG == 1) || (VALID_ROT_MAT_FLAG == 10) || (VALID_ROT_MAT_FLAG == 11))
		  { 
		    if(VERBOSE)
		      cout << "Video HARRIS Rotation Matrix: " << vidHARRISrotMat1 << endl;
		    vidHARRISanglesArr[0] = vidHARRIStheta_x1;
		    vidHARRISanglesArr[1] = vidHARRIStheta_y1;
		    vidHARRISanglesArr[2] = vidHARRIStheta_z1;
		  }
		else if((VALID_ROT_MAT_FLAG == 2) || (VALID_ROT_MAT_FLAG == 20) || (VALID_ROT_MAT_FLAG == 22))
		  { 
		    if(VERBOSE)
		      cout << "Video HARRIS Rotation Matrix: " << vidHARRISrotMat2 << endl;
		    vidHARRISanglesArr[0] = vidHARRIStheta_x2;
		    vidHARRISanglesArr[1] = vidHARRIStheta_y2;
		    vidHARRISanglesArr[2] = vidHARRIStheta_z2;
		  }		   
		else
                  {
                    cout << "Video HARRIS - NEITHER ROT MAT WORKS. Setting angles to 0,0,0. FLAG = " << VALID_ROT_MAT_FLAG << endl;
                    cout << "Theta x1,y1,z1 = " << vidHARRIStheta_x1 << "," << vidHARRIStheta_y1 << "," << vidHARRIStheta_z1 << endl;
                    cout << "Theta x2,y2,z2 = " << vidHARRIStheta_x2 << "," << vidHARRIStheta_y2 << "," << vidHARRIStheta_z2 << endl;
                    vidHARRISanglesArr[0] = 0;
                    vidHARRISanglesArr[1] = 0;
                    vidHARRISanglesArr[2] = 0;
		    HARRIS_EstValid = 0;
                    //exit(0);    
                  }


		Mat vidHARRISangles = Mat(3,1,DataType<double>::type,vidHARRISanglesArr);
		if((camDataType == LINUX0DEG)||(camDataType == WINDOWS0DEG))
		  {
		    rotatedVidHARRISangles = RcamtoV*vidHARRISangles;
		    rotatedVidHARRIStransl = RcamtoV*vidHARRIStranslMat;
		  }
		if((camDataType == LINUX17DEG)||(camDataType == WINDOWS17DEG))
		  {
		    rotatedVidHARRISangles = RcamtoV*Rneg17x*vidHARRISangles;
		    rotatedVidHARRIStransl = RcamtoV*Rneg17x*vidHARRIStranslMat;
		  }
		if((camDataType == LINUX30DEG)||(camDataType == WINDOWS30DEG))
		  {
		    rotatedVidHARRISangles = RcamtoV*Rneg30x*vidHARRISangles;
		    rotatedVidHARRIStransl = RcamtoV*Rneg30x*vidHARRIStranslMat;
		  }
		if((camDataType == LINUX68DEG)||(camDataType == WINDOWS68DEG))
		  {
		    rotatedVidHARRISangles = RcamtoV*Rneg68x*vidHARRISangles;
		    rotatedVidHARRIStransl = RcamtoV*Rneg68x*vidHARRIStranslMat;
		  }
		if((camDataType == BLENDER_DOWN)||(camDataType == BLENDER_FRONT))
		  {
		    rotatedVidHARRISangles = RcamtoV*vidHARRISangles;
		    rotatedVidHARRIStransl = RcamtoV*vidHARRIStranslMat;
		  }
		if(camDataType == GTRI_LAPTOP)
		  {
		    rotatedVidHARRISangles = RcamtoV*Rneg20x*vidHARRISangles;
		    rotatedVidHARRIStransl = RcamtoV*Rneg20x*vidHARRIStranslMat;
		  }	     
		if((camDataType == GOPRO_BRITNEY)||(camDataType == ICEFIN_FRONT)||(camDataType == ICEFIN_DOWN))
		  {
		    rotatedVidHARRISangles = RcamtoV*vidHARRISangles;
		    rotatedVidHARRIStransl = RcamtoV*vidHARRIStranslMat;
		  }

		//Multiply R and t by -1 to get vehicle-relative instead of world-relative

		if(ZERO_ROLLPITCH)
		  {
		    rotatedVidHARRISangles.at<double>(0,0) = 0;
		    rotatedVidHARRISangles.at<double>(0,1) = 0;
		  }

		rotatedVidHARRIStransl = rotatedVidHARRIStransl*-1;
		rotatedVidHARRISangles = rotatedVidHARRISangles*-1;

		//Check for large jumps - Bad estimate
		if(THRESHOLD_JUMPS)
		  {
		    if((abs(rotatedVidHARRISangles.at<double>(0,0)) > MAX_ROT)||
		       (abs(rotatedVidHARRISangles.at<double>(0,1)) > MAX_ROT)||
		       (abs(rotatedVidHARRISangles.at<double>(0,2)) > MAX_ROT))
		      {
			cout << "HARRIS Angles out of reasonable Range. Zeroed." << endl;
			cout << rotatedVidHARRISangles << endl;
			rotatedVidHARRISangles = Mat::zeros(3,1,CV_64F);
			HARRIS_EstValid = 0;
		      }
		  }

		//Get the sum angles
		vidHARRISsumAngles = vidHARRISsumAngles + rotatedVidHARRISangles;

		if(VERBOSE)
		  {
		    cout << "HARRIS VALID ROT MAT FLAG = " << VALID_ROT_MAT_FLAG << endl;
		    cout << "vidHARRISangles = " << vidHARRISangles << endl;
		    cout << "rotatedVidHARRISangles = " << rotatedVidHARRISangles << endl;
		    cout << "sumRotatedVidHARRISangles = " << vidHARRISsumAngles << endl;
		    cout << "Video HARRIS Translation: " << vidHARRIStranslMat << endl;
		    cout << "Video HARRIS R1=" << vidHARRISrotMat1 << endl;
		    cout << "Video HARRIS R2=" << vidHARRISrotMat2 << endl;
		    cout << "Video HARRIS Theta_x1 and _x2: " << vidHARRIStheta_x1 << " -- " << vidHARRIStheta_x2 << endl;
		    cout << "Video HARRIS Theta_y1 and _y2: " << vidHARRIStheta_y1 << " -- " << vidHARRIStheta_y2 << endl;
		    cout << "Video HARRIS Theta_z1 and _z2: " << vidHARRIStheta_z1 << " -- " << vidHARRIStheta_z2 << endl;
		  }

		if(DISPLAY_IMGS)
		  {
		    //------------Show Features--------------//

		    //Draw Matches:
		    drawMatches(prev_videoImgGray, vidHARRISkeypoints1, videoImgGray, vidHARRISkeypoints2, vidHARRISgood_matches, vidMatchesImg,Scalar::all(-1), Scalar::all(-1), vector<char>()); //, DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		    
		    //Draw Features:
		    for(int i1 = 0; i1 < vidHARRISgood_match_pts1.size(); i1++)
		      {
			Scalar color;
			if(vidHARRISmask.at<char>(0,i1))
			  color = Scalar(0,255,0); //Green
			else
			  color = Scalar(0,0,255); //Red
			circle(vidFeaturesImg, vidHARRISgood_match_pts1[i1], 4, color);
		      }
		  }
	      }

	    else // Found Less than 5 matches
	      {
		cout << "NO HARRIS MODEL FOUND" << endl;
		vidHARRISmodel = Mat(3,3,DataType<double>::type,essentialMatZero);
		vidHARRIStranslMat = Mat::zeros(3,1,CV_64F);
		HARRIS_EstValid = 0;
	      }
	  }
	else // Didn't Find Any Features
	  {
	    cout << "NO HARRIS MODEL FOUND" << endl;
	    vidHARRISmodel = Mat(3,3,DataType<double>::type,essentialMatZero);
	    vidHARRIStranslMat = Mat::zeros(3,1,CV_64F);
	    HARRIS_EstValid = 0;
	  }

	if(VERBOSE)
	  {
	    //cout << "Vid HARRIS Mask: " << endl << vidHARRISmask << endl;
	    cout << "Num HARRIS Matches: " << vidHARRISnumMatches << endl;
	    cout << "Num HARRIS Inliers: " << vidHARRISnumInliers << endl;
	    cout << "Video HARRIS Model: " << vidHARRISmodel << endl;
	  }

	if(DISPLAY_IMGS)
	  {
	    putText(vidFeaturesImg, "HARRIS", Point(10,25),FONT_HERSHEY_SIMPLEX,1,Scalar(255,255,255));
	    putText(vidMatchesImg, "HARRIS", Point(10,25),FONT_HERSHEY_SIMPLEX,1,Scalar(255,255,255));

	    imshow(vid_features_wnd, vidFeaturesImg);
	    imshow(vid_matches_wnd, vidMatchesImg);
	    if(STOP_BETWEEN_IMGS)
	      waitKey(0);
	    else
	      waitKey(10);
	  }
	if(WRITE_IMGS)
	  {
	    imwrite("HARRISvidFeaturesIMG.jpg", vidFeaturesImg);
	    imwrite("HARRISvidMatchesIMG.jpg", vidMatchesImg);
	  }


	//Find Global Rotation Matrix:
	//double tempAnglesArr[3] = {7, 11, 13};
	//Mat tempAngles = Mat(3,1,DataType<double>::type,tempAnglesArr);
	//cout << "sumAnglesGlobal_BEFORE" <<  vidOFsumAnglesGlobal << endl; 
	vidOFsumAnglesGlobal = vidOFsumAnglesGlobal + rotatedVidOFangles;
	vidSURFsumAnglesGlobal = vidSURFsumAnglesGlobal + rotatedVidSURFangles;
	vidSIFTsumAnglesGlobal = vidSIFTsumAnglesGlobal + rotatedVidSIFTangles;
	vidHARRISsumAnglesGlobal = vidHARRISsumAnglesGlobal + rotatedVidHARRISangles;
	if(VERBOSE)
	  {
	    cout << "rotatedAngles" <<  rotatedVidOFangles << endl; 
	    cout << "sumAnglesGlobal_AFTER" <<  vidOFsumAnglesGlobal << endl; 
	    cout << "vidOFsumAngles" <<  vidOFsumAngles << endl; 
	  }

	double phiOF = vidOFsumAnglesGlobal.at<double>(0,0)*CV_PI/180;
	double thetaOF = vidOFsumAnglesGlobal.at<double>(0,1)*CV_PI/180;
	double psiOF = vidOFsumAnglesGlobal.at<double>(0,2)*CV_PI/180;
	if(VERBOSE)
	  cout << "OF sumAnglesGlobal(r,p,y) = " << phiOF << "," << thetaOF << "," << psiOF << endl;

	double phiSURF = vidSURFsumAnglesGlobal.at<double>(0,0)*CV_PI/180;
	double thetaSURF = vidSURFsumAnglesGlobal.at<double>(0,1)*CV_PI/180;
	double psiSURF = vidSURFsumAnglesGlobal.at<double>(0,2)*CV_PI/180;

	double phiSIFT = vidSIFTsumAnglesGlobal.at<double>(0,0)*CV_PI/180;
	double thetaSIFT = vidSIFTsumAnglesGlobal.at<double>(0,1)*CV_PI/180;
	double psiSIFT = vidSIFTsumAnglesGlobal.at<double>(0,2)*CV_PI/180;

	double phiHARRIS = vidHARRISsumAnglesGlobal.at<double>(0,0)*CV_PI/180;
	double thetaHARRIS = vidHARRISsumAnglesGlobal.at<double>(0,1)*CV_PI/180;
	double psiHARRIS = vidHARRISsumAnglesGlobal.at<double>(0,2)*CV_PI/180;

	double rotGlobalzyxOF[3][3] = {cos(thetaOF)*cos(psiOF), sin(phiOF)*sin(thetaOF)*cos(psiOF)-cos(phiOF)*sin(psiOF), cos(phiOF)*sin(thetaOF)*cos(psiOF)+sin(phiOF)*sin(psiOF), 
				     cos(thetaOF)*sin(psiOF), sin(phiOF)*sin(thetaOF)*sin(psiOF)+cos(phiOF)*cos(psiOF), cos(phiOF)*sin(thetaOF)*sin(psiOF)-sin(phiOF)*cos(psiOF), 
				     -sin(thetaOF), sin(phiOF)*cos(thetaOF), cos(phiOF)*cos(thetaOF)};
	double rotGlobalzyxSURF[3][3] = {cos(thetaSURF)*cos(psiSURF), sin(phiSURF)*sin(thetaSURF)*cos(psiSURF)-cos(phiSURF)*sin(psiSURF), cos(phiSURF)*sin(thetaSURF)*cos(psiSURF)+sin(phiSURF)*sin(psiSURF), 
				     cos(thetaSURF)*sin(psiSURF), sin(phiSURF)*sin(thetaSURF)*sin(psiSURF)+cos(phiSURF)*cos(psiSURF), cos(phiSURF)*sin(thetaSURF)*sin(psiSURF)-sin(phiSURF)*cos(psiSURF), 
				     -sin(thetaSURF), sin(phiSURF)*cos(thetaSURF), cos(phiSURF)*cos(thetaSURF)};
	double rotGlobalzyxSIFT[3][3] = {cos(thetaSIFT)*cos(psiSIFT), sin(phiSIFT)*sin(thetaSIFT)*cos(psiSIFT)-cos(phiSIFT)*sin(psiSIFT), cos(phiSIFT)*sin(thetaSIFT)*cos(psiSIFT)+sin(phiSIFT)*sin(psiSIFT), 
				     cos(thetaSIFT)*sin(psiSIFT), sin(phiSIFT)*sin(thetaSIFT)*sin(psiSIFT)+cos(phiSIFT)*cos(psiSIFT), cos(phiSIFT)*sin(thetaSIFT)*sin(psiSIFT)-sin(phiSIFT)*cos(psiSIFT), 
				     -sin(thetaSIFT), sin(phiSIFT)*cos(thetaSIFT), cos(phiSIFT)*cos(thetaSIFT)};
	double rotGlobalzyxHARRIS[3][3] = {cos(thetaHARRIS)*cos(psiHARRIS), sin(phiHARRIS)*sin(thetaHARRIS)*cos(psiHARRIS)-cos(phiHARRIS)*sin(psiHARRIS), cos(phiHARRIS)*sin(thetaHARRIS)*cos(psiHARRIS)+sin(phiHARRIS)*sin(psiHARRIS), 
				     cos(thetaHARRIS)*sin(psiHARRIS), sin(phiHARRIS)*sin(thetaHARRIS)*sin(psiHARRIS)+cos(phiHARRIS)*cos(psiHARRIS), cos(phiHARRIS)*sin(thetaHARRIS)*sin(psiHARRIS)-sin(phiHARRIS)*cos(psiHARRIS), 
				     -sin(thetaHARRIS), sin(phiHARRIS)*cos(thetaHARRIS), cos(phiHARRIS)*cos(thetaHARRIS)};

	Mat RglobalOF = Mat(3,3,DataType<double>::type,rotGlobalzyxOF);
	Mat RglobalSURF = Mat(3,3,DataType<double>::type,rotGlobalzyxSURF);
	Mat RglobalSIFT = Mat(3,3,DataType<double>::type,rotGlobalzyxSIFT);
	Mat RglobalHARRIS = Mat(3,3,DataType<double>::type,rotGlobalzyxHARRIS);

	if(VERBOSE)
	  {
	    cout << "OF rotatedVidOFtransl=" << rotatedVidOFtransl << endl;
	    cout << "OF Global Add=" << RglobalOF*rotatedVidOFtransl << endl;
	  }

	vidOFsumTranslGlobal = vidOFsumTranslGlobal + RglobalOF*rotatedVidOFtransl;
	vidSURFsumTranslGlobal = vidSURFsumTranslGlobal + RglobalSURF*rotatedVidSURFtransl;
	vidSIFTsumTranslGlobal = vidSIFTsumTranslGlobal + RglobalSIFT*rotatedVidSIFTtransl;
	vidHARRISsumTranslGlobal = vidHARRISsumTranslGlobal + RglobalHARRIS*rotatedVidHARRIStransl;

	if(VERBOSE)
	  {
	    cout << "OF Global Angles=" << endl << vidOFsumAnglesGlobal << endl;
	    cout << "OF Global Transl=" << endl << vidOFsumTranslGlobal << endl;
	    cout << "SURF Global Angles=" << endl << vidSURFsumAnglesGlobal << endl;
	    cout << "SURF Global Transl=" << endl << vidSURFsumTranslGlobal << endl;
	    cout << "SIFT Global Angles=" << endl << vidSIFTsumAnglesGlobal << endl;
	    cout << "SIFT Global Transl=" << endl << vidSIFTsumTranslGlobal << endl;
	    cout << "HARRIS Global Angles=" << endl << vidHARRISsumAnglesGlobal << endl;
	    cout << "HARRIS Global Transl=" << endl << vidHARRISsumTranslGlobal << endl;
	  }

	//	cout << "globalAngles:" << phi << "," << theta << "," << psi << endl;
	//	cout << "inputAngles:" << tempAngles.at<double>(0,0) << "," << tempAngles.at<double>(0,1) << "," << tempAngles.at<double>(0,2) << endl;
	//	cout << "tempAnglesRot:" << tempAnglesRot.at<double>(0,0) << "," << tempAnglesRot.at<double>(0,1) << "," << tempAnglesRot.at<double>(0,2) << endl;
	

	//--------------------------------------------------------//
	//----------------Check For Small Numbers/Zeros-----------//
	int i2;
	for(i2=0; i2<3; i2++)
	  {
	    if(abs(vidOFanglesArr[i2]) < ZERO)
	      vidOFanglesArr[i2] = 0;
	    if(abs(vidSURFanglesArr[i2]) < ZERO)
	      vidSURFanglesArr[i2] = 0;
	    if(abs(vidSIFTanglesArr[i2]) < ZERO)
	      vidSIFTanglesArr[i2] = 0;
	    if(abs(vidHARRISanglesArr[i2]) < ZERO)
	      vidHARRISanglesArr[i2] = 0;
	    
	    if(abs(rotatedVidOFangles.at<double>(0,i2)) < ZERO)
	      rotatedVidOFangles.at<double>(0,i2) = 0;
	    if(abs(rotatedVidSURFangles.at<double>(0,i2)) < ZERO)
	      rotatedVidSURFangles.at<double>(0,i2) = 0;
	    if(abs(rotatedVidSIFTangles.at<double>(0,i2)) < ZERO)
	      rotatedVidSIFTangles.at<double>(0,i2) = 0;
	    if(abs(rotatedVidHARRISangles.at<double>(0,i2)) < ZERO)
	      rotatedVidHARRISangles.at<double>(0,i2) = 0;

	    if(abs(vidOFsumAngles.at<double>(0,i2)) < ZERO)
	      vidOFsumAngles.at<double>(0,i2) = 0;
	    if(abs(vidSURFsumAngles.at<double>(0,i2)) < ZERO)
	      vidSURFsumAngles.at<double>(0,i2) = 0;
	    if(abs(vidSIFTsumAngles.at<double>(0,i2)) < ZERO)
	      vidSIFTsumAngles.at<double>(0,i2) = 0;
	    if(abs(vidHARRISsumAngles.at<double>(0,i2)) < ZERO)
	      vidHARRISsumAngles.at<double>(0,i2) = 0;

	    if(abs(rotatedVidOFtransl.at<double>(0,i2)) < ZERO)
	      rotatedVidOFtransl.at<double>(0,i2) = 0;
	    if(abs(rotatedVidSURFtransl.at<double>(0,i2)) < ZERO)
	      rotatedVidSURFtransl.at<double>(0,i2) = 0;
	    if(abs(rotatedVidSIFTtransl.at<double>(0,i2)) < ZERO)
	      rotatedVidSIFTtransl.at<double>(0,i2) = 0;
	    if(abs(rotatedVidHARRIStransl.at<double>(0,i2)) < ZERO)
	      rotatedVidHARRIStransl.at<double>(0,i2) = 0;

	    if(abs(vidOFsumAnglesGlobal.at<double>(0,i2)) < ZERO)
	      vidOFsumAnglesGlobal.at<double>(0,i2) = 0;
	    if(abs(vidSURFsumAnglesGlobal.at<double>(0,i2)) < ZERO)
	      vidSURFsumAnglesGlobal.at<double>(0,i2) = 0;
	    if(abs(vidSIFTsumAnglesGlobal.at<double>(0,i2)) < ZERO)
	      vidSIFTsumAnglesGlobal.at<double>(0,i2) = 0;
	    if(abs(vidHARRISsumAnglesGlobal.at<double>(0,i2)) < ZERO)
	      vidHARRISsumAnglesGlobal.at<double>(0,i2) = 0;

	    if(abs(vidOFsumTranslGlobal.at<double>(0,i2)) < ZERO)
	      vidOFsumTranslGlobal.at<double>(0,i2) = 0;
	    if(abs(vidSURFsumTranslGlobal.at<double>(0,i2)) < ZERO)
	      vidSURFsumTranslGlobal.at<double>(0,i2) = 0;
	    if(abs(vidSIFTsumTranslGlobal.at<double>(0,i2)) < ZERO)
	      vidSIFTsumTranslGlobal.at<double>(0,i2) = 0;
	    if(abs(vidHARRISsumTranslGlobal.at<double>(0,i2)) < ZERO)
	      vidHARRISsumTranslGlobal.at<double>(0,i2) = 0;
	  }

	//--------------------------------------------------------//
	//----------------Print Results to File ------------------//
	if((prev_i != i)||PROCESS_00_FRAME) //Don't print first one where same image is used for i and i+1
	  {
	    outfile << prev_i << ";" << i << ";";
	    if(OPENCV_ROTS_TO_CSV)
	      outfile << vidOFanglesArr[0] << ";" << vidOFanglesArr[1] << ";" << vidOFanglesArr[2] << ";";
	    outfile << rotatedVidOFangles.at<double>(0,0) << ";" << rotatedVidOFangles.at<double>(0,1) << ";" << rotatedVidOFangles.at<double>(0,2) << ";" << vidOFsumAngles.at<double>(0,0) << ";" << vidOFsumAngles.at<double>(0,1) << ";" << vidOFsumAngles.at<double>(0,2) << ";";
	    if(OPENCV_ROTS_TO_CSV)
	      outfile << vidSURFanglesArr[0] << ";" << vidSURFanglesArr[1] << ";" << vidSURFanglesArr[2] << ";";
	    outfile << rotatedVidSURFangles.at<double>(0,0) << ";" << rotatedVidSURFangles.at<double>(0,1) << ";" << rotatedVidSURFangles.at<double>(0,2) << ";" << vidSURFsumAngles.at<double>(0,0) << ";" << vidSURFsumAngles.at<double>(0,1) << ";" << vidSURFsumAngles.at<double>(0,2) << ";";
	    if(OPENCV_ROTS_TO_CSV)
	      outfile << vidSIFTanglesArr[0] << ";" << vidSIFTanglesArr[1] << ";" << vidSIFTanglesArr[2] << ";";
	    outfile << rotatedVidSIFTangles.at<double>(0,0) << ";" << rotatedVidSIFTangles.at<double>(0,1) << ";" << rotatedVidSIFTangles.at<double>(0,2) << ";" << vidSIFTsumAngles.at<double>(0,0) << ";" << vidSIFTsumAngles.at<double>(0,1) << ";" << vidSIFTsumAngles.at<double>(0,2) << ";";
	    if(OPENCV_ROTS_TO_CSV)
	      outfile << vidHARRISanglesArr[0] << ";" << vidHARRISanglesArr[1] << ";" << vidHARRISanglesArr[2] << ";";
	    outfile << rotatedVidHARRISangles.at<double>(0,0) << ";" << rotatedVidHARRISangles.at<double>(0,1) << ";" << rotatedVidHARRISangles.at<double>(0,2) << ";" << vidHARRISsumAngles.at<double>(0,0) << ";" << vidHARRISsumAngles.at<double>(0,1) << ";" << vidHARRISsumAngles.at<double>(0,2) << ";";

	    //Output vehicle coordinate shifts
	    outfile << rotatedVidOFtransl.at<double>(0,0) << ";" << rotatedVidOFtransl.at<double>(0,1) << ";" << rotatedVidOFtransl.at<double>(0,2) << ";";
	    outfile << rotatedVidSURFtransl.at<double>(0,0) << ";" << rotatedVidSURFtransl.at<double>(0,1) << ";" << rotatedVidSURFtransl.at<double>(0,2) << ";";
	    outfile << rotatedVidSIFTtransl.at<double>(0,0) << ";" << rotatedVidSIFTtransl.at<double>(0,1) << ";" << rotatedVidSIFTtransl.at<double>(0,2) << ";";
	    outfile << rotatedVidHARRIStransl.at<double>(0,0) << ";" << rotatedVidHARRIStransl.at<double>(0,1) << ";" << rotatedVidHARRIStransl.at<double>(0,2) << ";";
	    
	    //Write out global sums:
	    outfile << vidOFsumTranslGlobal.at<double>(0,0) << ";" << vidOFsumTranslGlobal.at<double>(0,1) << ";" << vidOFsumTranslGlobal.at<double>(0,2) << ";";
	    outfile << vidOFsumAnglesGlobal.at<double>(0,0) << ";" << vidOFsumAnglesGlobal.at<double>(0,1) << ";" << vidOFsumAnglesGlobal.at<double>(0,2) << ";";

	    outfile << vidSURFsumTranslGlobal.at<double>(0,0) << ";" << vidSURFsumTranslGlobal.at<double>(0,1) << ";" << vidSURFsumTranslGlobal.at<double>(0,2) << ";";
	    outfile << vidSURFsumAnglesGlobal.at<double>(0,0) << ";" << vidSURFsumAnglesGlobal.at<double>(0,1) << ";" << vidSURFsumAnglesGlobal.at<double>(0,2) << ";";

	    outfile << vidSIFTsumTranslGlobal.at<double>(0,0) << ";" << vidSIFTsumTranslGlobal.at<double>(0,1) << ";" << vidSIFTsumTranslGlobal.at<double>(0,2) << ";";
	    outfile << vidSIFTsumAnglesGlobal.at<double>(0,0) << ";" << vidSIFTsumAnglesGlobal.at<double>(0,1) << ";" << vidSIFTsumAnglesGlobal.at<double>(0,2) << ";";

	    outfile << vidHARRISsumTranslGlobal.at<double>(0,0) << ";" << vidHARRISsumTranslGlobal.at<double>(0,1) << ";" << vidHARRISsumTranslGlobal.at<double>(0,2) << ";";
	    outfile << vidHARRISsumAnglesGlobal.at<double>(0,0) << ";" << vidHARRISsumAnglesGlobal.at<double>(0,1) << ";" << vidHARRISsumAnglesGlobal.at<double>(0,2) << ";";

	    outfile << vidOFnumCorners << ";" << vidOFnumMatches << ";" << vidOFnumInliers << ";";
	    outfile << vidSURFnumCorners << ";" << vidSURFnumMatches << ";" << vidSURFnumInliers << ";";
	    outfile << vidSIFTnumCorners << ";" << vidSIFTnumMatches << ";" << vidSIFTnumInliers << ";";
	    outfile << vidHARRISnumCorners << ";" << vidHARRISnumMatches << ";" << vidHARRISnumInliers << ";";

	    outfile << OF_EstValid << ";" << SURF_EstValid << ";" << SIFT_EstValid << ";" << HARRIS_EstValid << ";" << endl; 
	  }
	
	//--------------------------------------------------------//   
        //----------------Print Results to GTSAM File ------------//   
	if((camDataType==BLENDER_DOWN)||(camDataType==BLENDER_FRONT)) //Sim data - use OF
	  {
	    outfileCamGTSAM << prev_i/VIDEO_FPS << ";" << i/VIDEO_FPS << ";";
	    //outfileCamGTSAM << prev_i << ";" << i << ";";
	    outfileCamGTSAM << rotatedVidOFtransl.at<double>(0,0) << ";" << rotatedVidOFtransl.at<double>(0,1) << ";" << rotatedVidOFtransl.at<double>(0,2) << ";" << rotatedVidOFangles.at<double>(0,0) << ";" << rotatedVidOFangles.at<double>(0,1) << ";" << rotatedVidOFangles.at<double>(0,2) << ";" << vidOFnumCorners << ";" << vidOFnumMatches << ";" << vidOFnumInliers << ";";
	    outfileCamGTSAM << OF_EstValid << ";";  // << SURF_EstValid << ";" << SIFT_EstValid << ";" << HARRIS_EstValid << ";";
	    outfileCamGTSAM << endl; 
	  }
	else //Real data - use SURF
	  {
	    outfileCamGTSAM << prev_i/VIDEO_FPS << ";" << i/VIDEO_FPS << ";";
	    //outfileCamGTSAM << prev_i << ";" << i << ";";
	    outfileCamGTSAM << rotatedVidSURFtransl.at<double>(0,0) << ";" << rotatedVidSURFtransl.at<double>(0,1) << ";" << rotatedVidSURFtransl.at<double>(0,2) << ";" << rotatedVidSURFangles.at<double>(0,0) << ";" << rotatedVidSURFangles.at<double>(0,1) << ";" << rotatedVidSURFangles.at<double>(0,2) << ";" << vidSURFnumCorners << ";" << vidSURFnumMatches << ";" << vidSURFnumInliers << ";";
	    outfileCamGTSAM << SURF_EstValid << ";";  // << SURF_EstValid << ";" << SIFT_EstValid << ";" << HARRIS_EstValid << ";";
	    outfileCamGTSAM << endl; 
	  }
	//---------------------------------------------//        
	//Save Previous Images:
        prev_videoImgGray = videoImgGray.clone();
	prev_i = i;

	OF_EstValid = -1;
	SURF_EstValid = -1;
	SIFT_EstValid = -1;
	HARRIS_EstValid = -1;

	//--------------------------------------------------------//
	//--------------------------------------------------------//
	//--------------------------------------------------------//

	//-------------------------------------------------------

	//Display Images:
	if(DISPLAY_IMGS)
	  {
	    if(WRITE_FRAME_NUM)
	      {
		//Put Frame Number on image 
		char text[255];
		sprintf(text, "%d", i);
		cv::putText(videoImg, text, cvPoint(0,25), cv::FONT_HERSHEY_SIMPLEX,1,cv::Scalar::all(255));
	      }

	    imshow(video_wnd, videoImg);
	    imshow(video_undist_wnd, videoImgUndistort);
	    //imshow(vid_features_wnd, vidFeaturesImg);
	    //imshow(vid_matches_wnd, vidMatchesImgOF);
	    //waitKey(0);
	  }


	if(WRITE_VIDEO)
	  {
	    //Write output OF video frame---------------------------------------
	    Mat outVideoImg = Mat::zeros(videoImg.rows*2,videoImg.cols*2,CV_8UC3);
	    Mat tmp1, tmp2, tmp3, tmp4, tmp5;
	    
	    resize(videoImg, tmp1, Size(videoSize.width,videoSize.height));
	    Mat mapRoi1(outVideoImg, Rect(0, 0, videoImg.cols, videoImg.rows));
	    tmp1.copyTo(mapRoi1);
	    
	    cvtColor(videoImgUndistort,tmp5,COLOR_GRAY2BGR);
	    resize(tmp5, tmp2, Size(videoSize.width,videoSize.height));
	    Mat mapRoi2(outVideoImg, Rect(videoImg.cols, 0, videoImg.cols, videoImg.rows));
	    tmp2.copyTo(mapRoi2);

	    resize(vidMatchesImgOF, tmp3, Size(2*videoSize.width,videoSize.height));
	    Mat mapRoi3(outVideoImg, Rect(0, videoImg.rows, 2*videoImg.cols, videoImg.rows));
	    tmp3.copyTo(mapRoi3);
	    
	    outputVideoOF << outVideoImg;

	    //Write output SURF video frame----------------------------------
	    outVideoImg = Mat::zeros(videoImg.rows*2,videoImg.cols*2,CV_8UC3);
	    
	    resize(videoImg, tmp1, Size(videoSize.width,videoSize.height));
	    tmp1.copyTo(mapRoi1);
	    
	    cvtColor(videoImgUndistort,tmp5,COLOR_GRAY2BGR);
	    resize(tmp5, tmp2, Size(videoSize.width,videoSize.height));
	    tmp2.copyTo(mapRoi2);

	    resize(vidMatchesImgSURF, tmp3, Size(2*videoSize.width,videoSize.height));
	    tmp3.copyTo(mapRoi3);
	    
	    outputVideoSURF << outVideoImg;
	  }
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
	
     destroyWindow(video_wnd);
     destroyWindow(video_undist_wnd);
     destroyWindow(vid_features_wnd);
     destroyWindow(vid_matches_wnd);
     outfile.close();
     outfileCamGTSAM.close();

     return 0;
}
