#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <math.h>
#include <iostream>

using namespace cv;
using namespace std;

Mat img;
int height;
int width;

static void help()
{
    cout
    << "\nThis program illustrates the use of findContours and drawContours\n"
    << "The original image is put up along with the image of drawn contours\n"
    << "Usage:\n"
    << "./contours2 <img>\n"
    << "\nA trackbar is put up which controls the contour level from -3 to 3\n"
    << endl;
}

const int w = 500;
int levels = 3;

vector<vector<Point> > contours;
vector<Vec4i> hierarchy;

static void on_trackbar(int, void*)
{
    Mat cnt_img = Mat::zeros(height, width, CV_8UC3);
    int _levels = levels - 3;
    drawContours( cnt_img, contours, _levels <= 0 ? 3 : -1, Scalar(128,255,255),
                  3, CV_AA, hierarchy, std::abs(_levels) );

    imshow("contours", cnt_img);
}

const char* keys =
{
    "{1| |stuff.jpg|image for converting to a grayscale}"
};

int main( int argc, char** argv)
{
    help();
    CommandLineParser parser(argc, argv, keys);
    string inputImage = parser.get<string>("1");
    img = imread(inputImage.c_str(), 0);

    if(img.empty())
    {
        cout << "Could not read input image file: " << inputImage << endl;
        return -1;
    }

    height = img.rows;
    width = img.cols;

    //show the faces
    namedWindow( "image", 1 );
    imshow( "image", img );
    //Extract the contours so that
    vector<vector<Point> > contours0;
    findContours( img, contours0, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

    contours.resize(contours0.size());
    for( size_t k = 0; k < contours0.size(); k++ )
        approxPolyDP(Mat(contours0[k]), contours[k], 3, true);

    namedWindow( "contours", 1 );
    createTrackbar( "levels+3", "contours", &levels, 7, on_trackbar );

    on_trackbar(0,0);
    waitKey();

    return 0;
}
