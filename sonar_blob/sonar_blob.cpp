#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <stdlib.h>
#include <stdio.h>

using namespace cv;
using namespace std;

Mat src_img, gray_img, filter_img, binary_img, morph_img_tmp, morph_img, components_img, boxes_img;
RNG rng(12345);
int filter_type = 0;
int thresh_level = 100;
int thresh_type = 3;
int morph_type = 0;
int morph_size = 1;
int morph_iterations = 2;
const int max_filter_type = 2;
const int max_thresh_level = 255;
const int max_thresh_type = 4;
const int max_morph_type = 2;
const int max_morph_size = 5;
const int max_morph_iterations = 10;
const int max_BINARY_value = 255;

//Image Window Names
char input_window[] = "Input Image";
char gray_window[] = "Gray Image";
char filtered_window[] = "Filtered Image";
char thresh_window[] = "Thresholded Binary Image";
char erode_dilate_window[] = "Erode Dilate Image";
char components_window[] = "Connected Components Image";
char bounding_boxes_window[] = "Bounding Boxes Image";

//Trackbar Names
char filter_type_tbar[] = "Filtering Method: \n 0: None \n 1: Histogram Eq \n 2: Dynamic Thresholding";
char thresh_level_tbar[] = "Threshold Value";
char thresh_type_tbar[] = "Threshold Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted";
char morph_type_tbar[] = "Erode/Dilate Element:\n 0: Rect \n 1: Cross \n 2: Ellipse";
char morph_size_tbar[] = "Erode/Dilate Kernel size:\n 2n +1"; 
char morph_iterations_tbar[] =  "Erode/Dilate Iterations";


static void Components(int, void*)
{
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    findContours( morph_img, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );

    /// Approximate contours to polygons + get bounding rects and circles
    vector<vector<Point> > contours_poly( contours.size() );
    vector<Rect> boundRect( contours.size() );
    vector<Point2f>center( contours.size() );
    vector<float>radius( contours.size() );

    for( int i = 0; i < contours.size(); i++ )
      { approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
	boundRect[i] = boundingRect( Mat(contours_poly[i]) );
	//minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );
      }
    /// Draw polygonal contour + bonding rects + circles
    boxes_img = Mat::zeros( src_img.size(), CV_8UC3 );
    for( int i = 0; i< contours.size(); i++ )
      {
	Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
	drawContours( boxes_img, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
	rectangle( boxes_img, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
	//circle( drawing, center[i], (int)radius[i], color, 2, 8, 0 );
      }
    /// Show in a window
    imshow( bounding_boxes_window, boxes_img );

    components_img = Mat::zeros(src_img.size(), CV_8UC3);

    if( !contours.empty() && !hierarchy.empty() )
    {
        // iterate through all the top-level contours,
        // draw each connected component with its own random color
        int idx = 0;
        for( ; idx >= 0; idx = hierarchy[idx][0] )
        {
            Scalar color( (rand()&255), (rand()&255), (rand()&255) );
            drawContours( components_img, contours, idx, color, CV_FILLED, 8, hierarchy );
        }
    }

    imshow( components_window, components_img );
}

/**
 * @function Morphology
 */
void Morphology( int, void* )
{
  int morph_elem = 0;
  if( morph_type == 0 ){morph_elem  = MORPH_RECT; }
  else if( morph_type == 1 ){ morph_elem = MORPH_CROSS; }
  else if( morph_type == 2) { morph_elem = MORPH_ELLIPSE; }

  Mat element = getStructuringElement( morph_elem,
                       Size( 2*morph_size + 1, 2*morph_size+1 ),
                       Point( morph_size, morph_size ) );
  /// Apply the erosion operation
  erode( binary_img, morph_img_tmp, element, Point(-1,-1), morph_iterations );
  dilate(morph_img_tmp, morph_img, element, Point(-1,-1), morph_iterations );
  imshow( erode_dilate_window, morph_img );
  
  Components(0,0);
}

/**
 * @function Threshold
 */
void Threshold( int, void* )
{
  /* 0: Binary
     1: Binary Inverted
     2: Threshold Truncated
     3: Threshold to Zero
     4: Threshold to Zero Inverted
   */

  threshold( filter_img, binary_img, thresh_level, max_BINARY_value,thresh_type );

  imshow( thresh_window, binary_img );

  Morphology(0,0);
}

/**
 * @function Filter
 */
void Filter( int, void* )
{
 
  switch(filter_type){
  case 0:
    filter_img = gray_img.clone();
    break;  //No Filtering
  case 1:
    equalizeHist(gray_img, filter_img);
    break;
  case 2:
    adaptiveThreshold(gray_img, filter_img, max_BINARY_value, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY,3,0);  // Last two arguments are blocksize and Constant subtracted from mean
    break;
  default:
    break;
  }

  imshow( filtered_window, filter_img );

  Threshold(0,0);
}



static void help()
{
    cout << "\n This program demonstrates connected components and use of the trackbar\n"
             "Usage: \n"
             "  ./connected_components <image(stuff.jpg as default)>\n"
             "The image is converted to grayscale and displayed, another image has a trackbar\n"
             "that controls thresholding and thereby the extracted contours which are drawn in color\n";
}

int main( int argc, const char** argv )
{
    help();

    src_img = imread(argv[1]);

    if(src_img.empty())
    {
        cout << "Could not read input image file: " << argv[1] << endl;
        return -1;
    }

    //Create Image Windows
    namedWindow(input_window, WINDOW_AUTOSIZE);
    namedWindow(gray_window, WINDOW_AUTOSIZE);
    namedWindow(filtered_window, WINDOW_AUTOSIZE);
    namedWindow(thresh_window, WINDOW_AUTOSIZE);
    namedWindow(erode_dilate_window, WINDOW_AUTOSIZE);
    namedWindow(components_window, WINDOW_AUTOSIZE);
    namedWindow(bounding_boxes_window, WINDOW_AUTOSIZE);

    //Create Trackbars
    createTrackbar( filter_type_tbar,
                  filtered_window, &filter_type,
                  max_filter_type, Filter );

    createTrackbar( thresh_level_tbar,
                  thresh_window, &thresh_level,
                  max_thresh_level, Threshold );

    createTrackbar( thresh_type_tbar,
                  thresh_window, &thresh_type,
                  max_thresh_type, Threshold );

    createTrackbar( morph_type_tbar,
                  erode_dilate_window, &morph_type,
                  max_morph_type, Morphology );

    createTrackbar( morph_size_tbar,
                  erode_dilate_window, &morph_size,
                  max_morph_size, Morphology );

    createTrackbar( morph_iterations_tbar,
		  erode_dilate_window, &morph_iterations,
                  max_morph_iterations, Morphology );    

    //Show Original Image
    imshow(input_window, src_img );

    //Get Gray Image
    cvtColor(src_img, gray_img, COLOR_BGR2GRAY);
    imshow(gray_window, gray_img);

    //Default Start
    Filter(0,0);
    Threshold(0,0);
    Morphology(0,0);
    Components(0,0);

    waitKey(0);
    imwrite("output.png", boxes_img);
    return 0;
}
