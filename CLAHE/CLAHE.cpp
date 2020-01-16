#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <string>
#include <iostream>
#include <cstdlib>


using namespace cv;
using namespace std;
using cv::CLAHE;

int clipLimit = 0;
int maxClipLimit = 17;
int keyinput;
int fileNameLength = 0;
bool VIDEO = false;
Mat m, gray, dst;

int main( int argc, char** argv )
{
  if(argc < 2)
    {
      cout << "Usage: ./CLAHE <image_location>" << endl;
      return(0);
    }

  fileNameLength = strlen(argv[1]);
  if((argv[1][fileNameLength-3] == 'a')&&(argv[1][fileNameLength-2] == 'v')&&(argv[1][fileNameLength-1] == 'i'))
    VIDEO = true;

  Ptr<CLAHE> clahe = createCLAHE();
  namedWindow("GRAYSCALE",1);
  namedWindow("CLAHE",1);
  createTrackbar("Clip Limit", "CLAHE", &clipLimit, maxClipLimit);
  if(VIDEO)
    {
      VideoCapture cap(argv[1]);
      cap.read(m);
      while(1)
	{
	  cap.read(m);
	  cvtColor(m,gray,CV_BGR2GRAY);
	  imshow("GRAYSCALE",gray);
	  clahe->setClipLimit(clipLimit);
	  clahe->apply(gray,dst);
	  imshow("CLAHE",dst);
	  keyinput = waitKey(10);
	  if(keyinput == 27)
	    break;
	}
    }
  else 
    {
      m= imread(argv[1],CV_LOAD_IMAGE_GRAYSCALE); //input image
    
      imshow("GRAYSCALE",m);
      clahe->apply(m,dst);
      imshow("CLAHE",dst);

      for(;;)
	{
	  clahe->setClipLimit(clipLimit);
	  clahe->apply(m,dst);
	  imshow("CLAHE",dst);
	  keyinput = waitKey(1000);
	  if(keyinput == 27)
	    {
	      imwrite("./CLAHE_output.jpg",dst);
	      break;
	    }
	}
    }
}
