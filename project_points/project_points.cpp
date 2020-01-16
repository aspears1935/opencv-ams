#include <iostream>
#include <sstream>
#include <time.h>
#include <stdio.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

#define PI 3.14159265359

using namespace cv;
using namespace std;

std::vector<cv::Point3d> Generate3DPoints();
std::vector<cv::Point2f> Generate2DPoints();

double fovx;
double fovy;
double height;
double width;

int main(int argc, char* argv[])
{

  if(argc < 2)
    cout << "Usage: ./ProjectPoints <path_to_chess_image>" << endl;

  Mat src = imread(argv[1],CV_LOAD_IMAGE_GRAYSCALE);
  Size patternsize(6,6);
  vector<Point2f> corners;
  bool patternfound = findChessboardCorners(src,patternsize,corners,CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
  if(patternfound)
    cornerSubPix(src, corners, Size(11,11), Size(-1,-1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
  //drawChessboardCorners(src, patternsize, Mat(corners), patternfound);


//double cmat[3][3] = {{fx, 0, cx}, {0, fy, cy}, {0, 0, 1}};
  double carr[3][3] = {361.52, 0, 359.5, 0, 363.871, 269.5, 0, 0, 1};
  //float rarr[3][3] = {{1, 0, 0}, {0, 1, 0, 0, 0, 1}; //Identity matrix??
  //float tarr[3] = ;
  //double darr[5][1] = {-0.29454, 0.112814, 0.00269189, 0.002134487, -0.02799817};

  Mat cameraMatrix(3,3,DataType<double>::type,carr); //Intrinsics
  Mat rvec = Mat::zeros(3,1,DataType<double>::type); //Rodrigues Rotation Vector
  Mat rmat = Mat::eye(3,3,DataType<double>::type); //Rotation Matrix (Euler??)
  Mat tvec = Mat::zeros(3,1,DataType<double>::type);
  Mat distCoeffs = Mat::zeros(5,1,DataType<double>::type);//,darr);

  std::vector<Point2d> imagePoints;
  std::vector<Point3d> objectPoints = Generate3DPoints();
  std::vector<Point2f> object2DPoints = Generate2DPoints();
  std::vector<Point2d> projectedPoints;
  std::vector<Point3d> translated3DPoints;
  std::vector<Point3d> fov3DPoints;
  std::vector<Point2d> fov2DPoints;

  Size imageSize = src.size();
  height = imageSize.height;
  width = imageSize.width;
  cout << "Image Height: " << height << endl;
  cout << "Image Width" << width << endl;

  double apertureWidth = 0.0;
  double apertureHeight = 0.0;
  double focalLength;
  Point2d principalPoint;
  double aspectRatio;

  calibrationMatrixValues(cameraMatrix, imageSize, apertureWidth, apertureHeight, fovx, fovy, focalLength, principalPoint, aspectRatio);

  cout << "FOVX = " << fovx << endl << "FOVY" << fovy << endl;

  //Solve for rvec and tvec
  solvePnP(objectPoints, corners, cameraMatrix, distCoeffs, rvec, tvec);

  projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);
  Rodrigues(rvec, rmat); //Convert from rodrigues to euler??

  cout << "Cam Mat: " << endl << cameraMatrix << endl;
  cout << "Rot Mat: " << endl << rmat << endl;
  cout << "Tr Mat: " << endl << tvec << endl;
  cout << "Dist Coeffs: " << endl << distCoeffs << endl;

  //Generate FOV Border Tests and add to object points
  double fovx_border_xzratio = tan(fovx*(PI/180)/2);
  double fovy_border_yzratio = tan(fovy*(PI/180)/2);
  cout << "FOV X Ratio: " << fovx_border_xzratio << endl;
  cout << "FOV Y Ratio: " << fovy_border_yzratio << endl;
  fov3DPoints.push_back(Point3d(double(fovx_border_xzratio*10),0,double(10)));
  fov3DPoints.push_back(Point3d(double(-fovx_border_xzratio*10),0,double(10)));
  fov3DPoints.push_back(Point3d(0,double(fovy_border_yzratio*10),double(10)));
  fov3DPoints.push_back(Point3d(0,double(-fovy_border_yzratio*10),double(10)));
  fov3DPoints.push_back(Point3d(double(-fovx_border_xzratio*10),double(-fovy_border_yzratio*10),double(10)));
  fov3DPoints.push_back(Point3d(double(fovx_border_xzratio*10),double(-fovy_border_yzratio*10),double(10)));
  fov3DPoints.push_back(Point3d(double(-fovx_border_xzratio*10),double(fovy_border_yzratio*10),double(10)));
  fov3DPoints.push_back(Point3d(double(fovx_border_xzratio*10),double(fovy_border_yzratio*10),double(10)));
  projectPoints(fov3DPoints, Mat::zeros(3,1,DataType<double>::type), Mat::zeros(3,1,DataType<double>::type), cameraMatrix, distCoeffs, fov2DPoints);

  Mat H = findHomography(object2DPoints,corners);
  Mat cmat2, rmat2, tmat2;
  //  decomposeProjectionMatrix(H,cmat2,rmat2,tmat2);
  //cout << "Homog Cmat: " << endl << cmat2 << endl;
  //cout << "Homog Rmat: " << endl << rmat2 << endl;
  //cout << "Homog Tmat: " << endl << tmat2 << endl;
  cout << "Homog Mat: " << endl << H << endl;
  
  //Create 3D Homography Transformation Matrix 
  Mat transformMat = rmat.clone();// = Mat::zeros(4,4,DataType<double>::type);
  hconcat(transformMat, tvec, transformMat);
  double lastRowvec[4] = {0,0,0,1}; 
  Mat lastRow(1,4,DataType<double>::type,lastRowvec);
  vconcat(transformMat, lastRow, transformMat);
  cout << "TRANSFORM MAT: " << endl << transformMat << endl;

  //Create fake 3D Homography Transformation Matrix - debug
  //double fakeHomogVec[4][4] = {10,10,10,10, 10,10,10,10, 10,10,10,10, 0,0,0,1};
  //Mat transformMatFake(4,4,DataType<double>::type,fakeHomogVec);
  //cout << "Fake TRANSFORM MAT:" << endl << transformMatFake << endl;

  //Rotate and translate object points      
  perspectiveTransform(objectPoints,translated3DPoints,transformMat); //Has to be in homography coordinates
  
  Mat dst = src.clone();


  for(unsigned int i = 0; i < projectedPoints.size(); ++i)
    {
      circle(dst, projectedPoints[i], 8, Scalar(255,0,0));

      //      cout << "Object Point: " << objectPoints[i] << "  Projected To: " << projectedPoints[i] << endl;
      cout << "Object Point: " << objectPoints[i] << "  Projected To: " << translated3DPoints[i] << endl;
    }
  //Display fov edges
  for(int i=0; i<fov2DPoints.size(); ++i)
    circle(dst, fov2DPoints[i], 8, Scalar(255,255,255));

  namedWindow("Output",WINDOW_AUTOSIZE);
  imshow("Output", dst);
  waitKey(0);
    return 0;
}


std::vector<cv::Point2f> Generate2DPoints()
{
  std::vector<cv::Point2f> points;
  float squareSize = 26.2;
  for(int i = 0; i < 6; ++i)
    {
      for(int j = 0; j < 6; ++j)
	points.push_back(Point2d(float(j*squareSize), float(i*squareSize)));
    }

  return points;
}

std::vector<cv::Point3d> Generate3DPoints()
{
  std::vector<cv::Point3d> points;
  float squareSize = 26.2; //mm size of squares

  for(int i = 0; i < 6; ++i)
    {  
      for(int j = 0; j < 6; ++j)
	points.push_back(Point3d(double(j*squareSize), double(i*squareSize),0));
    }

  cout << points << endl;

  /*float x,y,z;
 
  x=0;y=0;z=0;
  points.push_back(cv::Point3d(x,y,z));
 
  x=0;y=0;z=100;
  points.push_back(cv::Point3d(x,y,z));
 
  x=0;y=0;z=1;
  points.push_back(cv::Point3d(x,y,z));
 
  x=1;y=0;z=0;
  points.push_back(cv::Point3d(x,y,z));
 
  x=0;y=1;z=0;
  points.push_back(cv::Point3d(x,y,z));
 
  x=-1;y=0;z=0;
  points.push_back(cv::Point3d(x,y,z));
 
  x=0;y=-1;z=0;
  points.push_back(cv::Point3d(x,y,z));
  */
  /*
  for(unsigned int i = 0; i < points.size(); ++i)
    {
    std::cout << points[i] << std::endl;
    }
  */
  return points;
}
