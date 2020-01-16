#include <iostream>
#include <sstream>
#include <time.h>
#include <stdio.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

int k1_max = 2001;
int k2_max = 2001;
int k3_max = 2001;
int p1_max = 2001;
int p2_max = 2001;
int k1 = 1001;
int k2 = 1001;
int k3 = 1001;
int p1 = 1001;
int p2 = 1001;

Mat grid;
Mat src;
Size imageSize;




void cv::initUndistortRectifyMap( InputArray _cameraMatrix, InputArray _distCoeffs,
				  InputArray _matR, InputArray _newCameraMatrix,
				  Size size, int m1type, OutputArray _map1, OutputArray _map2 )
{
 
  cout << "USING MY OWN initUndistorRectifyMap" << endl;
  Mat cameraMatrix = _cameraMatrix.getMat(), distCoeffs = _distCoeffs.getMat();
  Mat matR = _matR.getMat(), newCameraMatrix = _newCameraMatrix.getMat();

  if( m1type <= 0 )
    m1type = CV_16SC2;
  CV_Assert( m1type == CV_16SC2 || m1type == CV_32FC1 || m1type == CV_32FC2 );
  _map1.create( size, m1type );
  Mat map1 = _map1.getMat(), map2;
  if( m1type != CV_32FC2 )
    {
      _map2.create( size, m1type == CV_16SC2 ? CV_16UC1 : CV_32FC1 );
      map2 = _map2.getMat();
    }
  else
    _map2.release();

  Mat_<double> R = Mat_<double>::eye(3, 3);
  Mat_<double> A = Mat_<double>(cameraMatrix), Ar;

  if( newCameraMatrix.data )
    Ar = Mat_<double>(newCameraMatrix);
  else
    Ar = getDefaultNewCameraMatrix( A, size, true );

  if( matR.data )
    R = Mat_<double>(matR);

  if( distCoeffs.data )
    distCoeffs = Mat_<double>(distCoeffs);
  else
    {
      distCoeffs.create(8, 1, CV_64F);
      distCoeffs = 0.;
    }

  CV_Assert( A.size() == Size(3,3) && A.size() == R.size() );
  CV_Assert( Ar.size() == Size(3,3) || Ar.size() == Size(4, 3));
  Mat_<double> iR = (Ar.colRange(0,3)*R).inv(DECOMP_LU);
  const double* ir = &iR(0,0);

  double u0 = A(0, 2),  v0 = A(1, 2);
  double fx = A(0, 0),  fy = A(1, 1);

  CV_Assert( distCoeffs.size() == Size(1, 4) || distCoeffs.size() == Size(4, 1) ||
	     distCoeffs.size() == Size(1, 5) || distCoeffs.size() == Size(5, 1) ||
	     distCoeffs.size() == Size(1, 8) || distCoeffs.size() == Size(8, 1));

  if( distCoeffs.rows != 1 && !distCoeffs.isContinuous() )
    distCoeffs = distCoeffs.t();

  double k1 = ((double*)distCoeffs.data)[0];
  double k2 = ((double*)distCoeffs.data)[1];
  double p1 = ((double*)distCoeffs.data)[2];
  double p2 = ((double*)distCoeffs.data)[3];
  double k3 = distCoeffs.cols + distCoeffs.rows - 1 >= 5 ? ((double*)distCoeffs.data)[4] : 0.;
  double k4 = distCoeffs.cols + distCoeffs.rows - 1 >= 8 ? ((double*)distCoeffs.data)[5] : 0.;
  double k5 = distCoeffs.cols + distCoeffs.rows - 1 >= 8 ? ((double*)distCoeffs.data)[6] : 0.;
  double k6 = distCoeffs.cols + distCoeffs.rows - 1 >= 8 ? ((double*)distCoeffs.data)[7] : 0.;

  for( int i = 0; i < size.height; i++ )
    {
      float* m1f = (float*)(map1.data + map1.step*i);
      float* m2f = (float*)(map2.data + map2.step*i);
      short* m1 = (short*)m1f;
      ushort* m2 = (ushort*)m2f;
      double _x = i*ir[1] + ir[2], _y = i*ir[4] + ir[5], _w = i*ir[7] + ir[8];

      for( int j = 0; j < size.width; j++, _x += ir[0], _y += ir[3], _w += ir[6] )
        {
	  double w = 1./_w, x = _x*w, y = _y*w;
	  double x2 = x*x, y2 = y*y;
	  double r2 = x2 + y2, _2xy = 2*x*y;
	  double kr = (1 + ((k3*r2 + k2)*r2 + k1)*r2)/(1 + ((k6*r2 + k5)*r2 + k4)*r2);
	  double u = fx*(x*kr + p1*_2xy + p2*(r2 + 2*x2)) + u0;
	  double v = fy*(y*kr + p1*(r2 + 2*y2) + p2*_2xy) + v0;
	  
	  if(j==0 && i==0)
	    {
	      cout << "_x = " << _x << endl;
	      cout << "_y = " << _y << endl;
	      cout << "_w = " << _w << endl;
	      cout << "x = " << x << endl;
	      cout << "y = " << y << endl;
	      cout << "w = " << w << endl;
	      cout << "x2 = " << x2 << endl;
	      cout << "y2 = " << y2 << endl;
	      cout << "r2 = " << r2 << endl;
	      cout << "kr = " << kr << endl;
	      cout << "u = " << u << endl;
	      cout << "v = " << v << endl;
	    }

	  if( m1type == CV_16SC2 )
            {
	      int iu = saturate_cast<int>(u*INTER_TAB_SIZE);
	      int iv = saturate_cast<int>(v*INTER_TAB_SIZE);
	      m1[j*2] = (short)(iu >> INTER_BITS);
	      m1[j*2+1] = (short)(iv >> INTER_BITS);
	      m2[j] = (ushort)((iv & (INTER_TAB_SIZE-1))*INTER_TAB_SIZE + (iu & (INTER_TAB_SIZE-1)));
            }
	  else if( m1type == CV_32FC1 )
            {
	      m1f[j] = (float)u;
	      m2f[j] = (float)v;
	      //if(j==0)
	      //cout << "j=0 -> u=" << u << endl;
            }
	  else
            {
	      m1f[j*2] = (float)u;
	      m1f[j*2+1] = (float)v;
            }
        }
    }
}






static void trackbarCallback(int, void*)
{
  //double carr[3][3] = {361.52, 0, 359.5, 0, 363.871, 269.5, 0, 0, 1};
  //double carr[3][3] = {100, 0, 300, 0, 100, 300, 0, 0, 1};
  double carr[3][3] = {450.247, 0, 359.5, 0, 452.456, 269.5, 0, 0, 1};
  //double darr[5][1] = {-0.29454, 0.112814, 0.00269189, 0.002134487, -0.02799817};
  //double darr[5][1] = {-0.297356, 0.11297, 0.00309039, 0.00158565, -0.248698};
  double darr[5][1] = {-0.297356, 0.11297, 0, 0, -0.0248698};
  //darr[5][1] = {0, 0, 0, 0, 0};

  //darr[0][0] = (float)(k1-1001)/1000;
  //darr[1][0] = (float)(k2-1001)/1000;
  //darr[2][0] = (float)(p1-1001)/1000;
  //darr[3][0] = (float)(p2-1001)/1000;
  //darr[4][0] = (float)(k3-1001)/1000;

 
  Mat cameraMatrix(3,3,DataType<double>::type,carr); //Intrinsics                   

  Mat rvec = Mat::zeros(3,1,DataType<double>::type); //Rodrigues Rotation Vector     
  Mat rmat = Mat::eye(3,3,DataType<double>::type); //Rotation Matrix (Euler??)       
  Mat tvec = Mat::zeros(3,1,DataType<double>::type);
  Mat distCoeffs(5,1,DataType<double>::type,darr);//,darr)                           
  Mat view, rview, map1, map2, newCameraMatrix;

  Mat_<double> iR1 = cameraMatrix.colRange(0,3)*rmat;
  Mat_<double> iR = iR1.inv(DECOMP_LU);
  const double* ir = &iR(0,0);

  initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),cameraMatrix,
			  imageSize, CV_32FC1, map1, map2); //Was CV_16SC2

  //  cout << map1.at<short>(0,0) << "," << map1.at<short>(0,1)  << endl;
  //  cout << map1.at<short>(2,0) << "," << map1.at<short>(2,1)  << endl;
  //  cout << map1.at<short>(0,2) << "," << map1.at<short>(1,2)  << endl;
  cout << "iR1:" << endl << iR1 << endl;
  cout << "iR:" << endl << iR << endl;
  //Find Points Mat.at(y,x);
  cout << "(0,0) -> " << map1.at<float>(0,0) << ", " << map2.at<float>(0,0) << endl;
  //  cout << "(1,1) -> " << map1.at<float>(1,1) << "," << map2.at<float>(1,1) << endl;
  //cout << "(10,10) -> " << map1.at<float>(10,10) << "," << map2.at<float>(10,10) << endl;
  cout << "(100,100) -> " <<  map1.at<float>(100,100) << "," << map2.at<float>(100,100) << endl;
  cout << "(259,169) -> " <<  map1.at<float>(169,259) << "," << map2.at<float>(169,259) << endl;
  cout << "(359,269) -> " << map1.at<float>(269,359) << "," << map2.at<float>(269,359) << endl;
  cout << "(719,539) -> " << map1.at<float>(539,719) << "," << map2.at<float>(539,719) << endl;
  cout << "DISTORTION COEFFS:" << endl << distCoeffs << endl; //AMS            

  Mat rgrid;
  //remap(grid, rgrid, map1, map2, INTER_LINEAR);
  remap(src, rgrid, map1, map2, INTER_LINEAR);  

  imshow("Output", rgrid);
  waitKey(100);
}




int main(int argc, char** argv)
{

  src = imread(argv[1]);

  imageSize = src.size();
  double alpha;

  grid = Mat::zeros(540,720,CV_8U);

  for(int i=0; i<=13; i++)
    {
      line(grid, Point (0,i*60), Point (grid.cols,i*60), Scalar(255,255,255));
      line(grid, Point (i*60,0), Point (i*60,grid.rows), Scalar(255,255,255));
    }

  namedWindow("Grid");
  namedWindow("Output");
  imshow("Grid", grid);
  waitKey(100);

  createTrackbar("k1_track","Output", &k1, k1_max, trackbarCallback);
  createTrackbar("k2_track","Output", &k2, k2_max, trackbarCallback);
  createTrackbar("k3_track","Output", &k3, k3_max, trackbarCallback);
  createTrackbar("p1_track","Output", &p1, p1_max, trackbarCallback);
  createTrackbar("p2_track","Output", &p2, p2_max, trackbarCallback);

  //************************************************************/

  //double cmat[3][3] = {{fx, 0, cx}, {0, fy, cy}, {0, 0, 1}};
  double carr[3][3] = {361.52, 0, 359.5, 0, 363.871, 269.5, 0, 0, 1};
  //float rarr[3][3] = {{1, 0, 0}, {0, 1, 0, 0, 0, 1}; //Identity matrix??
  //float tarr[3] = ;
  double darr[5][1] = {-0.29454, 0.112814, 0.00269189, 0.002134487, -0.02799817};

  Mat cameraMatrix(3,3,DataType<double>::type,carr); //Intrinsics
  Mat rvec = Mat::zeros(3,1,DataType<double>::type); //Rodrigues Rotation Vector
  Mat rmat = Mat::eye(3,3,DataType<double>::type); //Rotation Matrix (Euler??)
  Mat tvec = Mat::zeros(3,1,DataType<double>::type);
  Mat distCoeffs(5,1,DataType<double>::type,darr);//,darr)


  
  Mat view, rview, map1, map2, newCameraMatrix;
        initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),cameraMatrix,
            imageSize, CV_16SC2, map1, map2);

	cout << "DISTORTION COEFFS:" << endl << distCoeffs << endl; //AMS

	Mat rgrid;
	remap(grid, rgrid, map1, map2, INTER_LINEAR);

	imshow("Output", rgrid);
	waitKey(0);

    return 0;
}
