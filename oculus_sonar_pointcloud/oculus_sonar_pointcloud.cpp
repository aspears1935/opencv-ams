/**
 * @file Threshold.cpp
 * @brief Sample code that shows how to use the diverse threshold options offered by OpenCV
 * @author OpenCV team
 */

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"
//#include "opencv2/videoio.hpp"
#include <iostream>
#include <string>
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
int blur_value = 0;
int blur_type = 0;
int const blur_max_value=10;
int const blur_max_type=4;
int morph_value = 0; //size
int morph_nerode = 0;  //number of erosions
int morph_ndilate = 0;  //number of dilations
int const max_morph_value=10;
int const max_morph_nerode = 5;
int const max_morph_ndilate = 5;

Mat src, src_gray, src_masked_gray, dst, output_gray, output_concat, blurred, thresholded, eroded, dilated;
Mat mask, mask_gray, mask_gray_vid, mask_bgra, output, output_bgra;
Mat output_concat_small;
const char* window_name = "Threshold Demo";
const char* trackbars_name = "Trackbars";

const char* trackbar_type = "Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted";
const char* trackbar_value = "Value";
const char* trackbar_blur_type = "Blur Type: \n 0: None \n 1: Avg \n 2: Gaussian \n 3: Median \n 4: Bilateral";
const char* trackbar_blur_value = "Blur Value";
const char* trackbar_nerode = "Number of Erosions";
const char* trackbar_ndilate = "Number of Dilations";
const char* trackbar_morph_value = "Morphological Elem Size";

VideoWriter outputVideo;

/**
 * @function on_threshold_trackbar
 */
static void on_threshold_trackbar( int, void* )
{
  cvtColor( src, src_gray, COLOR_BGR2GRAY ); // Convert the image to Gray
  src.copyTo(src_masked_gray, mask_gray);
  //Blur image to reduce noise
    int blur_kernel_size=blur_value*2+1; //must be odd and non-zero
    if(blur_type==0)
        src_masked_gray.copyTo(blurred);
    else if(blur_type==1)
        blur(src_masked_gray,blurred,Size(blur_kernel_size,blur_kernel_size),Point(-1,-1));
    else if(blur_type==2)
        GaussianBlur(src_masked_gray,blurred,Size(blur_kernel_size,blur_kernel_size),0,0);
    else if(blur_type==3)
        medianBlur(src_masked_gray,blurred,blur_kernel_size);
    else if(blur_type=4)
        bilateralFilter(src_masked_gray,blurred,blur_kernel_size,(blur_kernel_size)*2,(blur_kernel_size)/2);

    threshold( blurred, thresholded, threshold_value, threshold_max_binary_value, threshold_type );
    
    int morph_elem_size=morph_value*2+1;
    Mat morph_element=getStructuringElement(MORPH_RECT,Size(morph_elem_size,morph_elem_size),Point(morph_value,morph_value));
    
    //Eroding:
    thresholded.copyTo(eroded);
    if(morph_nerode > 0)
        for (int i = 0; i < morph_nerode; ++i)
        {
            Mat temp;
            erode(eroded, temp, morph_element);    
            temp.copyTo(eroded);
        }
    
    eroded.copyTo(dilated);
    if(morph_ndilate > 0)
        for (int i = 0; i < morph_ndilate; ++i)
        {
            Mat temp;
            dilate(dilated, temp, morph_element);    
            temp.copyTo(dilated);
        }

    //Morphological opening - erosion then dilation - good for removing small objects
    //  morphologyEx(thresholded, opened, MORPH_RECT, morph_element);
    //Morphological closing - dilation then erosion - good for removing small holes
    //  morphologyEx(opened, closed, MORPH_RECT, morph_element);   

    dilated.copyTo(output_gray);

    //Create concat output/input image
    hconcat(src_masked_gray,output_gray,output_concat);

    resize(output_concat,output_concat_small,Size(),0.5,0.5);
    
    imshow( window_name, output_concat_small );
    if (outputVideo.isOpened()) {
      outputVideo << output_concat_small; //output_gray;
    }
}

/**
 * @function main
 */
int main( int argc, char** argv )
{
    //! [load]
  //    String fileName("/media/aspears3/Data/Oculus_20191107_160554_20191106_220719.png"); // by default
  string filePath;
  String fileDir("/media/aspears3/Data/");
  //String fileName("Oculus_20191107_160554_20191106_220719");
  //String fileName("Oculus_20191107_160554_20191106_221103_output");
  String fileName("Oculus_20191107_160554_20191106_220627");
  String fileExt(".png");
  VideoCapture cap;
  bool isVideo = false; //Differentiate between video and image inputs

  //string filePath = fileDir+fileName+fileExt;
  
  if (argc > 1)
    {
      //fileName=argv[1];
      filePath = argv[1];
    }
  //string filePath = fileDir+fileName+fileExt;
  
  //Check file extension to determine file type (video vs image)
  string fileExtension=filePath.substr(filePath.find_last_of(".") + 1);
  if((fileExtension == "png")||(fileExtension == "jpg")) {
    isVideo=false;
    cout << "Found Image File" << endl;
  }
  else if(fileExtension == "avi") {
    isVideo=true;
    cout << "Found Video File" << endl;
  }
  else {
    cout << "INVALID FILE TYPE: " << fileExtension << endl;
    return -1;
  }
  
  //Read in first image
  if(isVideo) {
    cap = VideoCapture(filePath); 
    if(!cap.isOpened()){
      cout << "Error opening video file" << endl;
      return -1;
    }
    cap >> src;
    imwrite("vidFrame.png", src);

    Size S = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),(int) (cap.get(CAP_PROP_FRAME_HEIGHT)/2));
    cout << S << endl;
    cout << cap.get(CAP_PROP_FOURCC) << endl;
    outputVideo.open("output.avi", VideoWriter::fourcc('X','V','I','D'), cap.get(CAP_PROP_FPS), S, true);
    if (!outputVideo.isOpened()) {
      cout  << "Could not open the output video for write" << endl;
      return -1;
    }
  }
  else {
    src = imread(filePath, IMREAD_COLOR ); // Load an image
    if (src.empty())
      {
	cout << "Cannot read the image: " << filePath << std::endl;
	return -1;
      }
  }
  
  ////////////////////////////////////////////////
  //NOW MASK THE IMAGE///////////
  //Mat mask, mask_gray, mask_gray_vid, mask_bgra, output, output_bgra;
  mask = imread( "oculus_template_cleaned.png", IMREAD_COLOR ); // Load an image
  cvtColor( mask, mask_gray, COLOR_BGR2GRAY ); // Convert the image to Gray
  cvtColor( mask, mask_bgra, COLOR_BGR2BGRA ); // Convert the image to Gray
  cvtColor( src, src_gray, COLOR_BGR2GRAY ); // Convert the image to Gray

  //If this is a video, the mask needs to be 100 pixels shorter than the image inputs, 60 pix trimmed off top and 40 trimmed off bottom
  if(isVideo) {
    //copyMakeBorder(mask_gray, mask_gray_vid, 100, 0, 0, 0, BORDER_REPLICATE);
    Rect vidROI(0,60,1920,1060);
    mask_gray_vid=mask_gray(vidROI);
    mask_gray_vid.copyTo(mask_gray);
  }

  src.copyTo(output, mask_gray);
  src_gray.copyTo(src_masked_gray, mask_gray);

  //cout << "Mask Size: " << mask.size() << endl;
  //cout << "Vid Mask Size: " << mask_gray_vid.size() << endl;
  //cout << "Img Size: " << src.size() << endl;
  //imwrite("oculus_video_mask.png",mask_gray_vid);
  
  //    bitwise_not(mask_gray,maskInv);
  
  vector<Mat> bgrChannels(3);
  split(src, bgrChannels);
  vector<Mat> channels;
  channels.push_back(bgrChannels[0]);
  channels.push_back(bgrChannels[1]);
  channels.push_back(bgrChannels[2]);
  channels.push_back(mask_gray);
  //merge(channels, output_bgra);
  
  //Display image 
  //namedWindow("Masked Output", WINDOW_AUTOSIZE);
  //imshow("Masked Output", output);
  //waitKey(0);
  
  //string outputName=imageName+".png";
  //imwrite(outputName, output_bgra);
  
  //Create concat output/input image
  hconcat(src_masked_gray,src_masked_gray,output_concat);
  
  //Threshold image
  threshold_value=0;
  namedWindow(window_name,WINDOW_AUTOSIZE); //Create Window
  namedWindow(trackbars_name,WINDOW_AUTOSIZE); //Create Trackbar Window
  createTrackbar(trackbar_value,trackbars_name,&threshold_value,threshold_max_binary_value,on_threshold_trackbar);
  on_threshold_trackbar(threshold_value, 0);
  
  createTrackbar(trackbar_type, trackbars_name, &threshold_type, threshold_max_type, on_threshold_trackbar);
  on_threshold_trackbar(threshold_type, 0);
  
  createTrackbar(trackbar_blur_value, trackbars_name, &blur_value, blur_max_value, on_threshold_trackbar);
  on_threshold_trackbar(blur_value, 0);
  
  createTrackbar(trackbar_blur_type, trackbars_name, &blur_type, blur_max_type, on_threshold_trackbar);
  on_threshold_trackbar(blur_type, 0);
  
  createTrackbar(trackbar_nerode, trackbars_name, &morph_nerode, max_morph_nerode, on_threshold_trackbar);
  on_threshold_trackbar(morph_nerode, 0);
  
  createTrackbar(trackbar_ndilate, trackbars_name, &morph_ndilate, max_morph_ndilate, on_threshold_trackbar);
  on_threshold_trackbar(morph_ndilate, 0);
  
  createTrackbar(trackbar_morph_value, trackbars_name, &morph_value, max_morph_value, on_threshold_trackbar);
  on_threshold_trackbar(morph_value, 0);
  
  
  if(isVideo) {
    while(true){
      imshow("Frame", output_concat_small);//src);
      waitKey(30); //ms. Should change this to match the input video
      cap >> src;
      if(src.empty())
	break;
      on_threshold_trackbar(threshold_value,0);
    }
  }
  else {
    waitKey(0);     
    return 0;
  }
  
  //When finished with video, clean up
  cap.release();
  destroyAllWindows();
  return 0;
}
