/**
 * @file SURF_FlannMatcher
 * @brief SURF detector + descriptor + FLANN Matcher
 * @author A. Huaman
 */

#include <stdio.h>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "RANSAC.hpp"

using namespace cv;
using namespace std;

void readme();
void decomposeEssentialMat( InputArray _E, OutputArray _R1, OutputArray _R2, OutputArray _t );

/**
 * @function main
 * @brief Main function
 */
int main( int argc, char** argv )
{
  if( argc != 3 )
  { readme(); return -1; }

  Mat img_1 = imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE );
  Mat img_2 = imread( argv[2], CV_LOAD_IMAGE_GRAYSCALE );

  if( !img_1.data || !img_2.data )
  { std::cout<< " --(!) Error reading images " << std::endl; return -1; }

  //-- Step 1: Detect the keypoints using SURF Detector
  int minHessian = 400;

  SurfFeatureDetector detector( minHessian );

  std::vector<KeyPoint> keypoints_1, keypoints_2;

  detector.detect( img_1, keypoints_1 );
  detector.detect( img_2, keypoints_2 );

  //-- Step 2: Calculate descriptors (feature vectors)
  SurfDescriptorExtractor extractor;

  Mat descriptors_1, descriptors_2;

  extractor.compute( img_1, keypoints_1, descriptors_1 );
  extractor.compute( img_2, keypoints_2, descriptors_2 );

  //-- Step 3: Matching descriptor vectors using FLANN matcher
  FlannBasedMatcher matcher;
  std::vector< DMatch > matches;
  matcher.match( descriptors_1, descriptors_2, matches );

  double max_dist = 0; double min_dist = 100;

  //-- Quick calculation of max and min distances between keypoints
  for( int i = 0; i < descriptors_1.rows; i++ )
  { double dist = matches[i].distance;
    if( dist < min_dist ) min_dist = dist;
    if( dist > max_dist ) max_dist = dist;
  }

  printf("-- Max dist : %f \n", max_dist );
  printf("-- Min dist : %f \n", min_dist );

  //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
  //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
  //-- small)
  //-- PS.- radiusMatch can also be used here.
  std::vector< DMatch > good_matches;

  for( int i = 0; i < descriptors_1.rows; i++ )
  { if( matches[i].distance <= max(2*min_dist, 0.02) )
    { good_matches.push_back( matches[i]); }
  }

  //-- Draw only "good" matches
  Mat img_matches;
  drawMatches( img_1, keypoints_1, img_2, keypoints_2,
               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

  //AMS added code to find homography between images
  std::vector<Point2f> good_match_pts1;
  std::vector<Point2f> good_match_pts2;
  std::vector<Point2f> good_match_pts_diff;

  for( int i = 0; i < good_matches.size(); i++ )
    {
      //-- Get the keypoints from the good matches
      good_match_pts1.push_back( keypoints_1[ good_matches[i].queryIdx ].pt );
      good_match_pts2.push_back( keypoints_2[ good_matches[i].trainIdx ].pt );
      good_match_pts_diff.push_back(keypoints_2[good_matches[i].trainIdx].pt - keypoints_1[good_matches[i].queryIdx].pt);
    }

  double ransacReprojThreshold = 3; //Max reproj error for inliers
  double carr[3][3] = {361.52, 0, 359.5, 0, 363.871, 269.5, 0, 0, 1};
  Mat cameraMatrix(3,3,DataType<double>::type,carr);
  Mat H = findHomography(good_match_pts1, good_match_pts2, CV_RANSAC, ransacReprojThreshold);
  Mat fundamentalMatrix = findFundamentalMat(good_match_pts1, good_match_pts2, CV_FM_RANSAC, 3, 0.99);
  Mat cameraMatrix_transpose;
  transpose(cameraMatrix,cameraMatrix_transpose);
  Mat essentialMatrix = cameraMatrix_transpose*fundamentalMatrix*cameraMatrix;
  //Mat essentialMatrix2 = findEssentialMat(good_match_pts1, good_match_pts2, 362, Point2d(359.5,269.5),CV_RANSAC,0.99,3);
  Mat R1,R2, t;
  //recoverPose(essentialMatrix,good_match_pts1, good_match_pts2, 362, Point2d(359.5,269.5));
  decomposeEssentialMat(essentialMatrix,R1,R2,t);
  
  //The following is more for sonar and testing/debugging. - only takes 3 or 4 points 
  //Mat affineTform = getAffineTransform(good_match_pts1,good_match_pts2);
  //Mat perspectiveTform = getPerspectiveTransform(good_match_pts1,good_match_pts2);

  cout << "Homography: " << endl << H << endl;
  cout << "Fundamental Matrix: " << endl << fundamentalMatrix << endl;
  cout << "Essential Matrix: " << endl << essentialMatrix << endl;
  //  cout << "Essential Matrix From findEssentialMatrix(): " << endl << essentialMatrix2 << endl;

  cout << "Rotation Matrix: " << endl << R1 << endl << R2 << endl;
  cout << "Translation: " << endl << t << endl;

  cout << "Number of matches" << good_match_pts1.size() << endl;
  
  /*  for(int i = 0; i < (int)good_match_pts1.size(); i++)
    {
      cout << good_match_pts1[i] - good_match_pts2[i] << endl;
    }
  */

  //-- Show detected matches
  imshow( "Good Matches", img_matches );

  for( int i = 0; i < (int)good_matches.size(); i++ )
  { printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx ); }

  //Run RANSAC - AMS
  Mat model, mask;
  double RANSAC_reprojthresh = 1;
  double RANSAC_param = 0.99;
  int ok = estimateTranslation(good_match_pts_diff, model, mask, RANSAC_reprojthresh, RANSAC_param);
  cout << "OK" << endl;
  cout << "GOOD PTS DIFF: " << good_match_pts_diff << endl;
  cout << "RANSAC Model: " << model << endl;
  cout << "MASK: " << mask << endl;

  //Plot The shifts (diffs) in 2d
  Mat shifts_img = Mat::zeros(1000,1000,CV_8UC3);
  Point2f offsetPoint = Point2f(500,500);
  line(shifts_img, Point(0,500), Point(1000,500), Scalar(255,0,0));
  line(shifts_img, Point(500,0), Point(500,1000), Scalar(255,0,0));
  for( int i = 0; i < good_matches.size(); i++ )
    {
      Scalar color;
      if(mask.at<char>(0,i))
	color = Scalar(255,255,255);
      else
	color = Scalar(0,0,255);
      circle(shifts_img, good_match_pts_diff[i]*100 + offsetPoint, 5, color, 2);
    }
  Point2f modelPoint;
  modelPoint.x = model.at<double>(0,0);
  modelPoint.y = model.at<double>(0,1);
  cout << modelPoint << endl;
  circle(shifts_img, modelPoint*100 + offsetPoint, 5, Scalar(0,255,0), 2);
  imshow("Match Plots", shifts_img);

  waitKey(0);

  return 0;
}

/**
 * @function readme
 */
void readme()
{ std::cout << " Usage: ./SURF_FlannMatcher <img1> <img2>" << std::endl; }


/***FUNCTION decomposeEssentialMat - Not in opencv yet**/
void decomposeEssentialMat( InputArray _E, OutputArray _R1, OutputArray _R2, OutputArray _t )
{
  Mat E = _E.getMat().reshape(1, 3);
  CV_Assert(E.cols == 3 && E.rows == 3);

  Mat D, U, Vt;
  SVD::compute(E, D, U, Vt);

  if (determinant(U) < 0) U *= -1.;
  if (determinant(Vt) < 0) Vt *= -1.;

  Mat W = (Mat_<double>(3, 3) << 0, 1, 0, -1, 0, 0, 0, 0, 1);
  W.convertTo(W, E.type());

  Mat R1, R2, t;
  R1 = U * W * Vt;
  R2 = U * W.t() * Vt;
  t = U.col(2) * 1.0;

  R1.copyTo(_R1);
  R2.copyTo(_R2);
  t.copyTo(_t);
}
