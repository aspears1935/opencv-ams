//#include "/home/anthony/opencv_2.3.0/opencv/modules/calib3d/src/precomp.hpp"
#include "precomp.hpp"

#include <algorithm>
#include <iterator>
#include <limits>

using namespace cv;
using namespace std;

class TranslationEstimatorCallback : public PointSetRegistrator::Callback
{
public:
    int runKernel( InputArray _m1, InputArray _m2, OutputArray _model ) const
    {
        Mat m1 = _m1.getMat(), m2 = _m2.getMat();

        const Point2d* from = m1.ptr<Point2d>();
        const Point2d* to   = m2.ptr<Point2d> (); //WE DONT USE THIS.

	Mat modelMat = Mat::zeros(1,2,CV_64F);

	modelMat.at<double>(0,0) = from[0].x;
	modelMat.at<double>(0,1) = from[0].y;
	
	//cout << "modelMat=" << modelMat << endl;

	modelMat.copyTo(_model);
	Mat model = _model.getMat();

	//cout << "m1 in runKernel = " << m1 << endl;
	//cout << "MODEL from runKernel = " << model << endl;

        return 1;
    }

    void computeError( InputArray _m1, InputArray _m2, InputArray _model, OutputArray _err ) const
    {
        Mat m1 = _m1.getMat(), m2 = _m2.getMat(), model = _model.getMat();
        const Point2d* from = m1.ptr<Point2d>();
        const Point2d* to   = m2.ptr<Point2d>();
        const double* modelPtr = model.ptr<double>();

        int count = m1.checkVector(2);
        CV_Assert( count > 0 );

        _err.create(count, 1, CV_32F);
        Mat err = _err.getMat();
        float* errptr = err.ptr<float>();

        for(int i = 0; i < count; i++ )
        {
            const Point2d& f = from[i];
            //const Point2d& t = to[i];

	    double xshift = f.x - modelPtr[0];
	    double yshift = f.y - modelPtr[1];

	    //cout << "ERROR:" << xshift << "," << yshift << endl;

            errptr[i] = (float)std::sqrt(xshift*xshift + yshift*yshift);
        }
    }

    bool checkSubset( InputArray _ms1, InputArray _ms2, int count ) const
    {
        return true;
    }
};


int estimateTranslationNew(InputArray _from, InputArray _to,
                         OutputArray _out, OutputArray _inliers,
                         double param1, double param2)
{
    Mat from = _from.getMat(), to = _to.getMat();
    int count = from.checkVector(2);
    
    CV_Assert( count >= 0 && to.checkVector(2) == count );

    Mat dFrom, dTo;
    from.convertTo(dFrom, CV_64F);
    to.convertTo(dTo, CV_64F);
    dFrom = dFrom.reshape(2, count);
    dTo = dTo.reshape(2, count);

    const double epsilon = DBL_EPSILON;
    param1 = param1 <= 0 ? 3 : param1;
    param2 = (param2 < epsilon) ? 0.99 : (param2 > 1 - epsilon) ? 0.99 : param2;

    return createRANSACPointSetRegistrator(makePtr<TranslationEstimatorCallback>(), 1, param1, param2)->run(dFrom, dTo, _out, _inliers);
}

void testEstimateTranslationNew()
{
std::vector<Point2f> points1(6);
  std::vector<Point2f> points2(6);
  Point2f p1(100,100);
  Point2f p2(5,5);
  Point2f p3(2,2);
  Point2f p4(3,3);
  Point2f p5(4,4);
  Point2f p6(15,15);
  points1[0] = p1;
  points1[1] = p2;
  points1[2] = p3;
  points1[3] = p4;
  points1[4] = p5;
  points1[5] = p6;
  points2[0] = p1;
  points2[1] = p2;
  points2[2] = p3;
  points2[3] = p4;
  points2[4] = p5;
  points2[5] = p6;
  /*
  cv::Ptr<CvMat> points1_mat = cvCreateMat(points1.size(),2, CV_32F);
  cv::Ptr<CvMat> points2_mat = cvCreateMat(points2.size(),2, CV_32F);
 
  float* ptr1 = reinterpret_cast<float*>(points1_mat->data.ptr); 
  float* ptr2 = reinterpret_cast<float*>(points2_mat->data.ptr); 
  float* ptrVec = reinterpret_cast<float*>(points1.data());

  //ptr1[0] = p1.x;
  //cout << points1 << " " << ptr1[0] << " "  << endl;

    for(int i = 0; i<6; i += 2)
    {
      ptr1[i] = points1[i].x;
      ptr1[i+1] = points1[i].y;
      cout << ptr1[i] << "," << ptr1[i+1] << endl;
    }

  cout << "VEC = " << points1 << endl;
  //cout << "MAT = " << points1_mat << endl;

  cv::Ptr<CvMat> model = cvCreateMat(2,1,CV_32F);
  */
  //cv::TranslationEstimator().runKernel(points1_mat,points2_mat,model);
  

  Mat model;
  Mat mask;
  
  //cout << "POINTS 1: " << points1 << endl;

  int ok = estimateTranslationNew(points1,points1,model,mask,3,0.9);

  //cout << "MODEL: " << model << endl;
  //cout << "MASK: " << mask << endl;
  //cout << "POINTS1: " << points1 << endl;
  //cout << "POINTS2: " << points2 << endl;
  //cout << ok << endl;
}
