/**
 * @file Threshold.cpp
 * @brief Sample code that shows how to use the diverse threshold options offered by OpenCV
 * @author OpenCV team
 */

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <stdio.h>
#include <string.h>

using namespace cv;
using namespace std;
using std::cout;

/// Global variables

int threshold_value = 0;
int threshold_type = 3;
int const max_value = 255;
int const max_type = 4;
int const max_binary_value = 255;

Mat src, src_gray, dst;
const char* window_name = "Threshold Demo";

const char* trackbar_type = "Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted";
const char* trackbar_value = "Value";

//![Threshold_Demo]
/**
 * @function Threshold_Demo
 */
static void Threshold_Demo( int, void* )
{
    /* 0: Binary
     1: Binary Inverted
     2: Threshold Truncated
     3: Threshold to Zero
     4: Threshold to Zero Inverted
    */
    threshold( src_gray, dst, threshold_value, max_binary_value, threshold_type );
    imshow( window_name, dst );
}
//![Threshold_Demo]

/**
 * @function main
 */
int main( int argc, char** argv )
{
    //! [load]
  //    String imageName("/media/aspears3/Data/Oculus_20191107_160554_20191106_220719.png"); // by default
    String imageDir("/media/aspears3/Data/");
    String imageName("Oculus_20191107_160554_20191106_220719");
    String imageExt(".png");

    //string imagePath = imageDir+imageName+imageExt;

    if (argc > 1)
    {
      imageName=argv[1];
      //        imagePath = argv[1];
    }
    string imagePath = imageDir+imageName+imageExt;
    src = imread( samples::findFile( imagePath ), IMREAD_COLOR ); // Load an image

    if (src.empty())
    {
        cout << "Cannot read the image: " << imagePath << std::endl;
        return -1;
    }

    cvtColor( src, src_gray, COLOR_BGR2GRAY ); // Convert the image to Gray
    //! [load]
    
    //Get a pixel intensity value from the color source image
    //NOTE: grey background from the oculus screengrabs is 66,66,66
    Vec3b intensity = src.at<Vec3b>(200,200);
    unsigned blue = intensity.val[0];
    unsigned green = intensity.val[1];
    unsigned red = intensity.val[2];
    std::cout << blue << std::endl;
    std::cout << green << std::endl;
    std::cout << red << std::endl;

    //Draw circle around pixel location
    circle(src, Point(200,200), 10, Scalar(0,0,255), 1, LINE_8);
    
    //Set a pixel intensity value from the image
    intensity.val[0]=0;
    intensity.val[1]=0;
    intensity.val[2]=255;

    //Find masks for the background color
    Mat mask;
    inRange(src, Scalar(66,66,66), Scalar(66,66,66), mask);
    src.setTo(Scalar(0,0,0), mask);
    Mat maskInv;
    bitwise_not(mask,maskInv);
    src.setTo(Scalar(255,255,255), maskInv);
    
    //Display image with pixel location highlighted
    namedWindow("Selected Pixel", WINDOW_AUTOSIZE);
    imshow("Selected Pixel", src);
    waitKey(0);

    imwrite("output_oculus_template.png", src);

    
    ////////////////////////////////////////////////
    //NOW MASK THE IMAGE///////////
    src = imread( samples::findFile( imagePath ), IMREAD_COLOR ); // Load an image

    Mat mask_gray, mask_bgra, output, output_bgra;
    mask = imread( "oculus_template_cleaned.png", IMREAD_COLOR ); // Load an image
    cvtColor( mask, mask_gray, COLOR_BGR2GRAY ); // Convert the image to Gray
    cvtColor( mask, mask_bgra, COLOR_BGR2BGRA ); // Convert the image to Gray
    
    src.copyTo(output, mask_gray);

    //    bitwise_not(mask_gray,maskInv);

    vector<Mat> bgrChannels(3);
    split(src, bgrChannels);
    vector<Mat> channels;
    channels.push_back(bgrChannels[0]);
    channels.push_back(bgrChannels[1]);
    channels.push_back(bgrChannels[2]);
    channels.push_back(mask_gray);
    merge(channels, output_bgra);
	      
    //Display image 
    namedWindow("Masked Output", WINDOW_AUTOSIZE);
    imshow("Masked Output", output);
    waitKey(0);

    string outputName=imageName+".png";
    
    imwrite(outputName, output_bgra);
        
    return 0;
}
