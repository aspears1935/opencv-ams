#include <stdio.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string.h>
#include <cmath>

//#include <cv.h>
//#include <highgui.h>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/photo/photo.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

#define PI 3.14159265359

int main( int argc, char *argv[] )
{
Mat source=imread(argv[1]);
int alpha_=90., beta_=90., gamma_=90.;
int f_ = 677, dist_ = 500;
 int tx_ = 500, ty_ = 500;

Mat destination;

char wndname1[] = "Source: ";
char wndname2[] = "WarpPerspective: ";
string tbarname1 = "Alpha";
string tbarname2 = "Beta";
string tbarname3 = "Gamma";
string tbarname4 = "f";
string tbarname5 = "Distance";
 string tbarname6 = "X shift";
 string tbarname7 = "Y shift";
namedWindow(wndname1, 1);
namedWindow(wndname2, 1);
createTrackbar(tbarname1, wndname2, &alpha_, 180);
createTrackbar(tbarname2, wndname2, &beta_, 180);
createTrackbar(tbarname3, wndname2, &gamma_, 180);
createTrackbar(tbarname4, wndname2, &f_, 2000);
createTrackbar(tbarname5, wndname2, &dist_, 2000);
 createTrackbar(tbarname6, wndname2, &tx_, 1000);
 createTrackbar(tbarname7, wndname2, &ty_, 1000);


imshow(wndname1, source);
while(true) {
    double f, dist;
    double alpha, beta, gamma;
    alpha = ((double)alpha_ - 90.)*PI/180;
    beta = ((double)beta_ - 90.)*PI/180;
    gamma = ((double)gamma_ - 90.)*PI/180;
    f = (double) f_;
    dist = (double) dist_;

    Size taille = source.size();
    double w = (double)taille.width, h = (double)taille.height;

    // Projection 2D -> 3D matrix
    Mat A1 = (Mat_<double>(4,3) <<
        1, 0, -w/2,
        0, 1, -h/2,
        0, 0,    0,
        0, 0,    1);

    // Rotation matrices around the X,Y,Z axis
    Mat RX = (Mat_<double>(4, 4) <<
        1,          0,           0, 0,
        0, cos(alpha), -sin(alpha), 0,
        0, sin(alpha),  cos(alpha), 0,
        0,          0,           0, 1);

    Mat RY = (Mat_<double>(4, 4) <<
        cos(beta), 0, -sin(beta), 0,
                0, 1,          0, 0,
        sin(beta), 0,  cos(beta), 0,
                0, 0,          0, 1);

    Mat RZ = (Mat_<double>(4, 4) <<
        cos(gamma), -sin(gamma), 0, 0,
        sin(gamma),  cos(gamma), 0, 0,
        0,          0,           1, 0,
        0,          0,           0, 1);

    // Composed rotation matrix with (RX,RY,RZ)
    Mat R = RX * RY * RZ;

    // Translation matrix on the Z axis change dist will change the height
    Mat T = (Mat_<double>(4, 4) <<
        1, 0, 0, tx_-500,
        0, 1, 0, ty_-500,
        0, 0, 1, dist,
        0, 0, 0, 1);

    // Camera Intrisecs matrix 3D -> 2D
    Mat A2 = (Mat_<double>(3,4) <<
        f, 0, w/2, 0,
        0, f, h/2, 0,
        0, 0,   1, 0);

    // Final and overall transformation matrix
    Mat transfo = A2 * (T * (R * A1));

    // Apply matrix transformation
    warpPerspective(source, destination, transfo, taille, INTER_CUBIC | WARP_INVERSE_MAP);

    imshow(wndname2, destination);
    int key = waitKey(30);
    if(key == 's')
      imwrite("output.png",destination);
}
}
