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
int const threshold_max_value = 255;
int const threshold_max_type = 4;
int const threshold_max_binary_value = 255;

Mat src, src_gray, src_masked_gray, dst, output_gray;
const char* window_name = "Threshold Demo";

const char* trackbar_type = "Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted";
const char* trackbar_value = "Value";

/**
 * @function on_threshold_trackbar
 */
static void on_threshold_trackbar( int, void* )
{
    threshold( src_masked_gray, output_gray, threshold_value, threshold_max_binary_value, threshold_type );
    imshow( window_name, output_gray );
}

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
    src = imread(imagePath, IMREAD_COLOR ); // Load an image

    if (src.empty())
    {
        cout << "Cannot read the image: " << imagePath << std::endl;
        return -1;
    }

    ////////////////////////////////////////////////
    //NOW MASK THE IMAGE///////////
    Mat mask, mask_gray, mask_bgra, output, output_bgra;
    mask = imread( "oculus_template_cleaned.png", IMREAD_COLOR ); // Load an image
    cvtColor( mask, mask_gray, COLOR_BGR2GRAY ); // Convert the image to Gray
    cvtColor( mask, mask_bgra, COLOR_BGR2BGRA ); // Convert the image to Gray
    cvtColor( src, src_gray, COLOR_BGR2GRAY ); // Convert the image to Gray
    
    src.copyTo(output, mask_gray);
    src_gray.copyTo(src_masked_gray, mask_gray);

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


    //Threshold image
    threshold_value=0;
    namedWindow(window_name,WINDOW_AUTOSIZE); //Create Window
    createTrackbar(trackbar_value,window_name,&threshold_value,threshold_max_binary_value,on_threshold_trackbar);
    on_threshold_trackbar(threshold_value, 0);
    waitKey(0);

        
    return 0;
}
