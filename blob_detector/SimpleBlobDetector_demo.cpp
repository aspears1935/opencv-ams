#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	const char	* wndName = "Source image",
		        * wndNameGray = "Gray img", 
	                * wndNameOut = "Out";
	//* filename = "../data/images/eyebrows-smoothed.bmp";

	Mat src, gray, thresh, binary;
	Mat out;
	vector<KeyPoint> keyPoints;
	vector< vector <Point> > contours;
	vector< vector <Point> > approxContours;

	SimpleBlobDetector::Params params;
	params.minThreshold = 40;
	params.maxThreshold = 60;
	params.thresholdStep = 5;

	params.minArea = 100; 
	params.minConvexity = 0.3;
	params.minInertiaRatio = 0.01;

	params.maxArea = 8000;
	params.maxConvexity = 10;

	params.filterByColor = false;
	params.filterByCircularity = false;

	namedWindow( wndNameOut, CV_GUI_NORMAL );

	src = imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE );
	line( src, Point(0, src.rows-1), Point( src.cols-1, src.rows-1 ), Scalar::all(255) );

	SimpleBlobDetector blobDetector( params );
	blobDetector.create("SimpleBlob");

	for(;;)
	{
		blobDetector.detect( src, keyPoints );
		//blobDetector.detectEx( src, keyPoints, contours );
		drawKeypoints( src, keyPoints, out, CV_RGB(0,255,0), DrawMatchesFlags::DEFAULT);
		approxContours.resize( contours.size() );

		for( int i = 0; i < contours.size(); ++i )
		{
			approxPolyDP( Mat(contours[i]), approxContours[i], 4, 1 );
			drawContours( out, contours, i, CV_RGB(rand()&255, rand()&255, rand()&255) );
			drawContours( out, approxContours, i, CV_RGB(rand()&255, rand()&255, rand()&255) );
		}
		cout << "Keypoints " << keyPoints.size() << endl;
	
		imshow( wndNameOut, out );
		if(waitKey(0) == 27)
		  break;
	}
}
