
#include <stdio.h>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"


using namespace cv;
using namespace std;


int main(int argc, char** argv)
{

    cv::Mat src1clr = cv::imread(argv[1]);
    cv::Mat src2clr = cv::imread(argv[2]);
    cv::Mat src3clr;
    cv::Mat src1, src2, src3;
    
    //If a third arg:
    if(argc > 3)
      src3clr = imread(argv[3]);

    // show the image
    cv::namedWindow("Image", CV_WINDOW_AUTOSIZE);
    cv::moveWindow("Image", 0, 0);
    //cv::imshow("Image", src1);

    cout << "Image Width = " << src1.cols << endl;
    cout << "Image Height = " << src1.rows << endl;

    //Create gray CLAHE images:
    int clipLimit = 4;
    Ptr<CLAHE> clahe = createCLAHE();
    clahe->setClipLimit(clipLimit);
    cvtColor(src1clr, src1, COLOR_BGR2GRAY);
    cvtColor(src2clr, src2, COLOR_BGR2GRAY);
    cvtColor(src3clr, src3, COLOR_BGR2GRAY);
    clahe->apply(src1,src1);
    clahe->apply(src2,src2);
    clahe->apply(src3,src3);

    //Show Images:
    cv::imshow("Image",src3);

    //Compute HOG for IMG1
    cv::HOGDescriptor hog;
    vector<float> descriptors1;
    vector<Point> locs1;
    hog.compute(src1, descriptors1,Size(32,32),Size(0,0),locs1);

    cout << "Num Descriptors: " <<  descriptors1.size() << endl;

    Mat Hogfeat1;
    Hogfeat1.create(descriptors1.size(),1,CV_32F);
    for(int i=0; i<descriptors1.size(); i++)
      {
	Hogfeat1.at<float>(i,0) = descriptors1.at(i);
      }

    //Compute HOG for IMG2
    vector<float> descriptors2;
    vector<Point> locs2;
    hog.compute(src2, descriptors2,Size(32,32),Size(0,0),locs2);

    Mat Hogfeat2;
    Hogfeat2.create(descriptors2.size(),1,CV_32F);
    for(int i=0; i<descriptors2.size(); i++)
      {
	Hogfeat2.at<float>(i,0) = descriptors2.at(i);
      }

    //Compute HOG for IMG3
    Mat Hogfeat3;
    vector<float> descriptors3;
    vector<Point> locs3;
    if(argc > 3)
      {
	hog.compute(src3, descriptors3,Size(32,32),Size(0,0),locs3);

	Hogfeat3.create(descriptors3.size(),1,CV_32F);
	for(int i=0; i<descriptors3.size(); i++)
	  {
	    Hogfeat3.at<float>(i,0) = descriptors3.at(i);
	  }
      }

    //This is for comparing the HOG features of two images without using any SVM 
    //(It is not an efficient way but useful when you want to compare only few or two images)
    //Simple distance
    //Consider you have two hog feature vectors for two images Hogfeat1 and Hogfeat2 and those are same size.
    double distance=0;
    for(int i=0;i<Hogfeat1.rows;i++)
      {
	distance += abs(Hogfeat1.at<float>(i,0) - Hogfeat2.at<float>(i,0));
      }

    cout << "Distance: " << distance << endl;

    float threshold = 100;
    if(distance < threshold)
      cout<<"Two images are of same class"<<endl;
    else
      cout<<"Two images are of different class"<<endl;
    
    //cout << "HOG DESCRIPTOR:" << descriptors << endl;
    cout << "Descriptor Size in Floats: " << hog.getDescriptorSize() << endl;
    cv::waitKey(0);


    /*******************************************************************/
    /****************** SVM CODE ***************************************/
    int pos_count = 1;
    int neg_count = 1;
    Mat labels( pos_count + neg_count, 1, CV_32FC1, Scalar(-1.0) );
    labels.rowRange( 0, pos_count ) = Scalar( 1.0 );

    CvSVM svm;
    CvSVMParams params;
    //params.C = c;
    params.svm_type = CvSVM::C_SVC;
    params.kernel_type = CvSVM::LINEAR;
    params.term_crit = cvTermCriteria( CV_TERMCRIT_ITER, 10000, 1e-6 );

    Mat descriptors1Mat_t(descriptors1, true);
    Mat descriptors2Mat_t(descriptors2, true);
    Mat descriptors3Mat_t(descriptors3, true);
    
    Mat descriptors1Mat;
    Mat descriptors2Mat;
    Mat descriptors3Mat;

    transpose(descriptors1Mat_t, descriptors1Mat);
    transpose(descriptors2Mat_t, descriptors2Mat);
    if(argc > 3)
      transpose(descriptors3Mat_t, descriptors3Mat);
    
    cout << descriptors1.size() << endl;
    cout << descriptors1Mat.cols << endl;
    
    Mat feat_matrix( pos_count + neg_count, descriptors1Mat.cols, descriptors1Mat.type() );

    cout << feat_matrix.size() << endl;
    
    for( int i = 0; i < pos_count; i++ )
      descriptors1Mat.copyTo( feat_matrix.rowRange(i, i+1) );
    for( int i = 0; i < neg_count; i++ )
      descriptors2Mat.copyTo( feat_matrix.rowRange(i + pos_count, i + pos_count + 1) );

    cout << "Start training SVM" << endl;
    cout << "Labels" << labels << endl;

    svm.train( feat_matrix, labels, Mat(), Mat(), params );
    svm.save("svmOutput.svm");

    int resultSVM;
    if(argc > 3)
      resultSVM = svm.predict(descriptors3Mat);

    cout << "RESULT:" << resultSVM << endl;
    
    return 0;
}
