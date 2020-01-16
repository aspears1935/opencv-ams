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

double camArr[3][3] = {677.694, 0, 319.5, 0, 676.564, 239.5, 0, 0, 1};

 int main( int argc, char *argv[] )
 {

      if (argc < 3) {
 	  printf("usage: ./sonar_opencv <image1-file> <image2-file>\n");
 	  exit(-1);
      }

      char img1FileName[256];
      char img2FileName[256];
      strcpy(img1FileName, argv[1]);
      strcpy(img2FileName, argv[2]);

      char vid_features_wnd[] = "Video Features Window";
      char vid_matches_wnd[] = "Video Matches Window";
      char video_wnd[] = "Video Window";

      namedWindow(vid_features_wnd,1);
      namedWindow(vid_matches_wnd,1);
      namedWindow(video_wnd,1);
      
      Mat img1 = imread(img1FileName,0); //0 = grayscale
      Mat img2 = imread(img2FileName,0);
      cv::Mat vidFeaturesImg, vidMatchesImg;


      cout << "Image Size = " << img1.cols << "," << img1.rows << endl;

      //-----------------------------------------------------------
      //SURF Detector:
      int minHessian = 400;
      SurfFeatureDetector vidSURFdetector(minHessian);
      SurfDescriptorExtractor vidSURFextractor;
      FlannBasedMatcher vidSURFmatcher;

	//SIFT Detector:
	cv::Ptr< cv::FeatureDetector > vidSIFTdetector = FeatureDetector::create("SIFT");	
	cv::Ptr< cv::DescriptorExtractor > vidSIFTextractor = DescriptorExtractor::create("SIFT");
	FlannBasedMatcher vidSIFTmatcher;

	//Create CLAHE:
	int clipLimit = 6;
	Ptr<CLAHE> clahe = createCLAHE();
	clahe->setClipLimit(clipLimit);


	//----------------------------------------------------------------------
	//--------------------Initialize Camera Detection ----------------------
	//----------------------------------------------------------------------

	Mat cameraMatrix = Mat(3,3,DataType<double>::type,camArr);
	Mat cameraMatrix_transpose;
	transpose(cameraMatrix,cameraMatrix_transpose);

	double fx = cameraMatrix.at<double>(0,0);
	double fy = cameraMatrix.at<double>(1,1);
	Point2d princ_pt = Point2d(cameraMatrix.at<double>(0,2),cameraMatrix.at<double>(1,2));

	cout << "New Cam Mat: " << cameraMatrix << endl;
	cout << "princ_pt: " << princ_pt << endl;

       
	//Apply CLAHE:
       clahe->apply(img1,img1);
       clahe->apply(img2,img2);

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
	//vidMatchesImg = prev_videoImgGray.clone();

	//Corner Detection:
	goodFeaturesToTrack( img1, vidOFcorners1, vidOFmax_corners, vidOFqualityLevel, vidOFminDistance, Mat(), vidOFblockSize, vidOFuseHarrisDetector, vidOF_k);

        if(vidOFcorners1.size() > 0)
	  {

	    //Calculate Corners Subpixel Accuracy:
	    Size vidOFsubPixWinSize(10,10);
	    cornerSubPix(img1, vidOFcorners1, vidOFsubPixWinSize, Size(-1,-1), vidOFtermcrit);

	    //Lucas Kanade Pyramid Algorithm:        
	    vector<uchar> vidOFstatus;
	    vector<float> vidOFerr;
	    //Mat vidOFpyrLKImg = videoImgGray.clone();

	    calcOpticalFlowPyrLK(img1, img2, vidOFcorners1, vidOFcorners2, vidOFstatus, vidOFerr, vidOFwinSize, 7, vidOFtermcrit, 0, 0.001);

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
	    if(vidOFgoodMatches1.size() > 0) //If found some matches
	      {
		//Find Fundamental and Essential Matrices
		double vidOF_RANSAC_reprojthresh = 1;
		double vidOF_RANSAC_param = 0.99;

		Mat vidOFfundamentalMatrix = findFundamentalMat(vidOFgoodMatches1, vidOFgoodMatches2, CV_FM_LMEDS, vidOF_RANSAC_reprojthresh, vidOF_RANSAC_param,vidOFmask);
		Mat temp_mask1;
		Mat vidOFessentialMatrix1 = findEssentialMatNew(vidOFgoodMatches1, vidOFgoodMatches2, fx, fy, princ_pt, RANSAC, vidOF_RANSAC_param, vidOF_RANSAC_reprojthresh,temp_mask1); //ADD MASK LATER
		//cout << "Essential Mat 1 = " << vidOFessentialMatrix1 << endl;
		Mat vidOFrotMat;
		Mat vidOFtranslMat;
		
		recoverPoseNew(vidOFessentialMatrix1, vidOFgoodMatches1, vidOFgoodMatches2, vidOFrotMat, vidOFtranslMat, fx, fy, princ_pt, temp_mask1);
		
		double theta_x = atan2(vidOFrotMat.at<double>(2,1),vidOFrotMat.at<double>(2,2))*180/CV_PI;
	  	double theta_y = atan2(-vidOFrotMat.at<double>(2,0),sqrt(vidOFrotMat.at<double>(2,1)*vidOFrotMat.at<double>(2,1) + vidOFrotMat.at<double>(2,2)*vidOFrotMat.at<double>(2,2)))*180/CV_PI;
		double theta_z = atan2(vidOFrotMat.at<double>(1,0),vidOFrotMat.at<double>(0,0))*180/CV_PI;  
		
		//cout << "Rotation: " << vidOFrotMat << endl;
		//cout << "Translation: " << vidOFtranslMat << endl;
		//cout << "Thetax: " << theta_x << endl;
		cout << "Thetay: " << theta_y << endl;
		//cout << "Thetaz: " << theta_z << endl;
	
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
		//Show Features:                                               
		cvtColor(img1, vidFeaturesImg, CV_GRAY2BGR); //Get copy of gray img to mark features      
		//drawMatches(prev_sonarImgGray, sonOFkeypoints1, sonarImgGray, sonOFkeypoints2, sonOFgood_matches, matchesImg,Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		//cout << vidOFcorners1.size() << endl;

		//Draw Feature Matches:
		vidNewMatchesImg = Mat(img2.rows,2*img2.cols,CV_8U);
		Point2f vidOFimg2offset = Point2f(img2.cols,0);
		Mat vid_roi1(vidNewMatchesImg,Rect(0,0,img2.cols,img2.rows));
		img1.copyTo(vid_roi1);
		Mat vid_roi2(vidNewMatchesImg,Rect(img2.cols,0,img2.cols,img2.rows));
		img2.copyTo(vid_roi2);
		cvtColor(vidNewMatchesImg, vidMatchesImg, CV_GRAY2BGR); //To Color

		//Draw Inliers and matches:
		for(int i1 = 0; i1 < vidOFgoodMatches1.size(); i1++)
		  {
		    Scalar color;
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
	imshow(vid_features_wnd, vidFeaturesImg);
	imshow(vid_matches_wnd, vidMatchesImg);
	waitKey(0);
	
	
	//------------ SURF Feature Detection -----------------
	cvtColor(img1, vidFeaturesImg, CV_GRAY2BGR); //Get copy of gray img to mark features      
	vidMatchesImg = img1.clone();
	Mat vidSURFmodel, vidSURFmask;
	float vidSURFnumInliers = 0;
	float vidSURFnumMatches = 0;
	std::vector<KeyPoint> vidSURFkeypoints_1, vidSURFkeypoints_2;
	vidSURFdetector.detect( img1, vidSURFkeypoints_1 );
	vidSURFdetector.detect( img2, vidSURFkeypoints_2 );
	//SURF Calculate descriptors (feature vectors):                             
	Mat vidSURFdescriptors_1, vidSURFdescriptors_2;
	vidSURFextractor.compute( img1, vidSURFkeypoints_1, vidSURFdescriptors_1 );
	vidSURFextractor.compute( img2, vidSURFkeypoints_2, vidSURFdescriptors_2 );

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
	      { if( vidSURFmatches[i1].distance <= max(2*vidSURFmin_dist, 0.02) )
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

	    if(vidSURFgood_matches.size() > 0)
	      {
		//vidSURFok = estimateTranslationNew(vidSURFgood_match_pts_diff, vidSURFgood_match_pts_diff, vidSURFmodel, vidSURFmask, vidSURF_RANSAC_reprojthresh, vidSURF_RANSAC_param);
		Mat vidSURFfundamentalMatrix = findFundamentalMat(vidSURFgood_match_pts1, vidSURFgood_match_pts2, CV_FM_LMEDS, vidSURF_RANSAC_reprojthresh, vidSURF_RANSAC_param, vidSURFmask);

		//cout << "SURF FUND MAT SIZE = " << vidSURFfundamentalMatrix << endl;
                Mat temp_mask1;
                Mat vidSURFessentialMatrix1 = findEssentialMatNew(vidSURFgood_match_pts1, vidSURFgood_match_pts2, fx, fy, princ_pt, RANSAC, vidSURF_RANSAC_param, vidSURF_RANSAC_reprojthresh,temp_mask1); //ADD MASK LATER                                                  
                //cout << "Essential Mat 1 = " << vidOFessentialMatrix1 << endl;     
                Mat vidSURFrotMat;
                Mat vidSURFtranslMat;

		recoverPoseNew(vidSURFessentialMatrix1, vidSURFgood_match_pts1, vidSURFgood_match_pts2, vidSURFrotMat, vidSURFtranslMat, fx, fy, princ_pt, temp_mask1);

		double theta_x = atan2(vidSURFrotMat.at<double>(2,1),vidSURFrotMat.at<double>(2,2))*180/CV_PI;
		double theta_y = atan2(-vidSURFrotMat.at<double>(2,0),sqrt(vidSURFrotMat.at<double>(2,1)*vidSURFrotMat.at<double>(2,1) + vidSURFrotMat.at<double>(2,2)*vidSURFrotMat.at<double>(2,2)))*180/CV_PI;
                double theta_z = atan2(vidSURFrotMat.at<double>(1,0),vidSURFrotMat.at<double>(0,0))*180/CV_PI;

		double camtoVarr[3][3] = {0, 0, 1, 1, 0, 0, 0, 1, 0};
                Mat RcamtoV = Mat(3,3,DataType<double>::type,camtoVarr);
                cout << "RcamtoV = " << endl << RcamtoV << endl;

       		Mat vidSURFrotMatNew = vidSURFrotMat*RcamtoV;
		//Mat vidSURFrotMatNew = vidSURFrotMat.clone();

                //Mat vidOFrotMatNew = Rvehtocam*vidOFrotMat*Rcamtoveh;

		double vidSURFtheta_xNew = atan2(vidSURFrotMatNew.at<double>(2,1),vidSURFrotMatNew.at<double>(2,2))*180/CV_PI;
                double vidSURFtheta_yNew = atan2(-vidSURFrotMatNew.at<double>(2,0),sqrt(vidSURFrotMatNew.at<double>(2,1)*vidSURFrotMatNew.at<double>(2,1) + vidSURFrotMatNew.at<double>(2,2)*vidSURFrotMatNew.at<double>(2,2)))*180/CV_PI;
                double vidSURFtheta_zNew = atan2(vidSURFrotMatNew.at<double>(1,0),vidSURFrotMatNew.at<double>(0,0))*180/CV_PI;

                cout << "Rotation: " << vidSURFrotMatNew << endl;
                cout << "Translation: " << vidSURFtranslMat << endl;
                cout << "Thetax: " << theta_x << "--" << vidSURFtheta_xNew << endl; 
                cout << "Thetay: " << theta_y << "--" << vidSURFtheta_yNew << endl;
                cout << "Thetaz: " << theta_z << "--" << vidSURFtheta_zNew << endl;

/*		double xyztheta_x = atan2(-vidSURFrotMat.at<double>(1,2),vidSURFrotMat.at<double>(2,2))*180/CV_PI;
		double xyztheta_y = atan2(vidSURFrotMat.at<double>(0,2),sqrt(vidSURFrotMat.at<double>(1,2)*vidSURFrotMat.at<double>(1,2) + vidSURFrotMat.at<double>(2,2)*vidSURFrotMat.at<double>(2,2)))*180/CV_PI;
                double xyztheta_z = atan2(-vidSURFrotMat.at<double>(0,1),vidSURFrotMat.at<double>(0,0))*180/CV_PI;

                cout << "xyzRightThetax: " << xyztheta_x << endl;  
                cout << "xyzRightThetay: " << xyztheta_y << endl;
                cout << "xyzRightThetaz: " << xyztheta_z << endl;
		*/


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

		//Draw Features: 
		drawMatches(img1, vidSURFkeypoints_1, img2, vidSURFkeypoints_2, vidSURFgood_matches, vidMatchesImg,Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
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
	imshow(vid_features_wnd, vidFeaturesImg);
	imshow(vid_matches_wnd, vidMatchesImg);
	//waitKey(0);

	//---------------video SIFT Feature Detection ------------------//
	cvtColor(img1, vidFeaturesImg, CV_GRAY2BGR); //Get copy of gray img to mark features      
	vidMatchesImg = img1.clone();
        Mat vidSIFTmodel, vidSIFTmask;
	float vidSIFTnumInliers = 0;
	float vidSIFTnumMatches = 0;
	std::vector<KeyPoint> vidSIFTkeypoints_1, vidSIFTkeypoints_2;
        vidSIFTdetector->detect( img1, vidSIFTkeypoints_1 );
        vidSIFTdetector->detect( img2, vidSIFTkeypoints_2 );
        //SURF Calculate descriptors (feature vectors):                             
        Mat vidSIFTdescriptors_1, vidSIFTdescriptors_2;
        vidSIFTextractor->compute( img1, vidSIFTkeypoints_1, vidSIFTdescriptors_1 );
        vidSIFTextractor->compute( img2, vidSIFTkeypoints_2, vidSIFTdescriptors_2 );

	if((!vidSIFTdescriptors_1.empty()) && (!vidSIFTdescriptors_2.empty()))
	  {
	    //SURF Matching descriptor vectors using FLANN matcher                  
	    std::vector< DMatch > vidSIFTmatches;
	    vidSIFTmatcher.match( vidSIFTdescriptors_1, vidSIFTdescriptors_2, vidSIFTmatches );
	    double vidSIFTmax_dist = 0; double vidSIFTmin_dist = 100;
	    //-- Quick calculation of max and min distances between keypoints            
	    for( int i1 = 0; i1 < vidSIFTdescriptors_1.rows; i1++ )
	      { double dist = vidSIFTmatches[i1].distance;
		if( dist < vidSIFTmin_dist ) vidSIFTmin_dist = dist;
		if( dist > vidSIFTmax_dist ) vidSIFTmax_dist = dist;
	      }
	    //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,  or a small arbitary value ( 0.02 ) in the event that min_dist is very small)   //-- PS.- radiusMatch can also be used here. 
	    std::vector< DMatch > vidSIFTgood_matches;
	    for( int i1 = 0; i1 < vidSIFTdescriptors_1.rows; i1++ )
	      { if( vidSIFTmatches[i1].distance <= max(2*vidSIFTmin_dist, 0.02) )
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
	 
	    if(vidSIFTgood_matches.size() > 0)
	      {
		Mat vidSIFTfundamentalMatrix = findFundamentalMat(vidSIFTgood_match_pts1, vidSIFTgood_match_pts2, CV_FM_LMEDS, vidSIFT_RANSAC_reprojthresh, vidSIFT_RANSAC_param,vidSIFTmask);


                Mat temp_mask1;
                Mat vidSIFTessentialMatrix1 = findEssentialMatNew(vidSIFTgood_match_pts1, vidSIFTgood_match_pts2, fx, fy, princ_pt, RANSAC, vidSIFT_RANSAC_param, vidSIFT_RANSAC_reprojthresh,temp_mask1); //ADD MASK LATER                                                  
                //cout << "Essential Mat 1 = " << vidOFessentialMatrix1 << endl;     
                Mat vidSIFTrotMat;
                Mat vidSIFTtranslMat;

		recoverPoseNew(vidSIFTessentialMatrix1, vidSIFTgood_match_pts1, vidSIFTgood_match_pts2, vidSIFTrotMat, vidSIFTtranslMat, fx, fy, princ_pt, temp_mask1);

		double theta_x = atan2(vidSIFTrotMat.at<double>(2,1),vidSIFTrotMat.at<double>(2,2))*180/CV_PI;
		double theta_y = atan2(-vidSIFTrotMat.at<double>(2,0),sqrt(vidSIFTrotMat.at<double>(2,1)*vidSIFTrotMat.at<double>(2,1) + vidSIFTrotMat.at<double>(2,2)*vidSIFTrotMat.at<double>(2,2)))*180/CV_PI;
                double theta_z = atan2(vidSIFTrotMat.at<double>(1,0),vidSIFTrotMat.at<double>(0,0))*180/CV_PI;

                cout << "Rotation: " << vidSIFTrotMat << endl;    
                cout << "Translation: " << vidSIFTtranslMat << endl;        
                cout << "Thetax: " << theta_x << endl;                             
                cout << "Thetay: " << theta_y << endl;
                cout << "Thetaz: " << theta_z << endl;

		double xyztheta_x = atan2(-vidSIFTrotMat.at<double>(1,2),vidSIFTrotMat.at<double>(2,2))*180/CV_PI;
		double xyztheta_y = atan2(vidSIFTrotMat.at<double>(0,2),sqrt(vidSIFTrotMat.at<double>(1,2)*vidSIFTrotMat.at<double>(1,2) + vidSIFTrotMat.at<double>(2,2)*vidSIFTrotMat.at<double>(2,2)))*180/CV_PI;
                double xyztheta_z = atan2(-vidSIFTrotMat.at<double>(0,1),vidSIFTrotMat.at<double>(0,0))*180/CV_PI;

                cout << "xyzRightThetax: " << xyztheta_x << endl;  
                cout << "xyzRightThetay: " << xyztheta_y << endl;
                cout << "xyzRightThetaz: " << xyztheta_z << endl;

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
		
		//Draw Features:                
		drawMatches(img1, vidSIFTkeypoints_1, img2, vidSIFTkeypoints_2, vidSIFTgood_matches, vidMatchesImg,Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);                      
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
        imshow(vid_features_wnd, vidFeaturesImg);    
	imshow(vid_matches_wnd, vidMatchesImg);
	waitKey(0);

	//-------------------------- HARRIS/SIFT Detection ---------------//
	//-- create detector and descriptor --
	// if you want it faster, take e.g. FAST or ORB
	Mat vidHARRISmodel, vidHARRISmask;
	float vidHARRISnumMatches = 0;
	float vidHARRISnumInliers = 0;

	cvtColor(img1, vidFeaturesImg, CV_GRAY2BGR); //Get copy of gray img to mark features
	vidMatchesImg = img1.clone();

	cv::Ptr<cv::FeatureDetector> vidHARRISdetector = cv::FeatureDetector::create("HARRIS"); 
	cv::Ptr<cv::DescriptorExtractor> vidHARRISdescriptor = cv::DescriptorExtractor::create("SIFT"); 

	// detect keypoints
	std::vector<cv::KeyPoint> vidHARRISkeypoints1, vidHARRISkeypoints2;
	vidHARRISdetector->detect(img1, vidHARRISkeypoints1);
	vidHARRISdetector->detect(img2, vidHARRISkeypoints2);

	// extract features
	cv::Mat vidHARRISdesc1, vidHARRISdesc2;
	vidHARRISdescriptor->compute(img1, vidHARRISkeypoints1, vidHARRISdesc1);
	vidHARRISdescriptor->compute(img2, vidHARRISkeypoints2, vidHARRISdesc2);

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
	      { if( vidHARRISmatches[i1].distance <= max(2*vidHARRISmin_dist, 0.02) )
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
	   
	    if(vidHARRISgood_matches.size() > 0)
	      {
		Mat vidHARRISfundamentalMatrix = findFundamentalMat(vidHARRISgood_match_pts1, vidHARRISgood_match_pts2, CV_FM_LMEDS, vidHARRIS_RANSAC_reprojthresh, vidHARRIS_RANSAC_param,vidHARRISmask);

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

		//Draw Features:
		drawMatches(img1, vidHARRISkeypoints1, img2, vidHARRISkeypoints2, vidHARRISgood_matches, vidMatchesImg,Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
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
	imshow(vid_features_wnd, vidFeaturesImg);
	imshow(vid_matches_wnd, vidMatchesImg);
	waitKey(0);
	

	//--------------------------------------------------------------
	//--------------------------------------------------------------
	//--------------------------------------------------------------

	//Display Images:
	imshow(video_wnd, img1);
	//imshow(son_features_wnd, sonFeaturesImg);
	//imshow(vid_features_wnd, vidFeaturesImg);
	waitKey(0);

	
     destroyWindow(video_wnd);
	
     return 0;
}
