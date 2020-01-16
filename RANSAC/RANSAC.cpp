//#include "precomp.hpp"
#include "_modelest.h"
//#include <algorithm>
//#include <iterator>
//#include <limits>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

namespace cv
{

  class TranslationEstimator : public CvModelEstimator2
{
public:
    TranslationEstimator() : CvModelEstimator2(1, cvSize(2, 1), 1) {}
  virtual int runKernel( const CvMat* m1, const CvMat* m2, CvMat* model );
  //virtual bool runRANSAC(const CvMat* m1, const CvMat* m2, CvMat* model,CvMat* mask0, double reprojThreshold, double confidence, int maxIters );
protected:
    virtual void computeReprojError( const CvMat* m1, const CvMat* m2, const CvMat* model, CvMat* error );

};

}
/*
bool TranslationEstimator::runRANSAC( const CvMat* m1, const CvMat* m2, CvMat* model,
				   CvMat* mask0, double reprojThreshold,
				   double confidence, int maxIters )
{

  bool result = false;
  cv::Ptr<CvMat> mask = cvCloneMat(mask0);
  cv::Ptr<CvMat> models, err, tmask;
  cv::Ptr<CvMat> ms1, ms2;

  int iter, niters = maxIters;
  int count = m1->rows*m1->cols, maxGoodCount = 0;

  CV_Assert( CV_ARE_SIZES_EQ(m1, m2) && CV_ARE_SIZES_EQ(m1, mask) );

  if( count < modelPoints )
    return false;

  models = cvCreateMat( modelSize.height*maxBasicSolutions, modelSize.width, CV_64FC1 );
  err = cvCreateMat( 1, count, CV_32FC1 );
  tmask = cvCreateMat( 1, count, CV_8UC1 );

  if( count > modelPoints )
    {
      ms1 = cvCreateMat( 1, modelPoints, m1->type );
      ms2 = cvCreateMat( 1, modelPoints, m2->type );
    }
  else
    {
      niters = 1;
      ms1 = cvCloneMat(m1);
      ms2 = cvCloneMat(m2);
    }

  for( iter = 0; iter < niters; iter++ )
    {
      int i, goodCount, nmodels;
      if( count > modelPoints )
        {
	  bool found = getSubset( m1, m2, ms1, ms2, 300 );
	  if( !found )
            {
	      if( iter == 0 )
		return false;
	      break;
            }
        }

      nmodels = runKernel( ms1, ms2, models );
      if( nmodels <= 0 )
	continue;
      for( i = 0; i < nmodels; i++ )
        {
	  CvMat model_i;
	  cvGetRows( models, &model_i, i*modelSize.height, (i+1)*modelSize.height );

	  goodCount = findInliers( m1, m2, &model_i, err, tmask, reprojThreshold );
	  cout << "AMS GOOD COUNT " << goodCount << endl;
	  if( goodCount > MAX(maxGoodCount, modelPoints-1) )
            {
	      std::swap(tmask, mask);
	      cv::Ptr<CvMat> temp1, temp2; //AMS
	      temp1 = &model_i;
	      cout << "AMS ABOUT TO COPY " << endl;
	      cout << "ROWS: " << temp1->rows << " AND  " << model->rows << endl;
	      cout << "COLS: " << temp1->cols << " AND " << model->cols << endl;
	      cvCopy( &model_i, model );
	      
	      cout << "AMS DONE COPY " << endl;
	      maxGoodCount = goodCount;
	      niters = cvRANSACUpdateNumIters( confidence,
					       (double)(count - goodCount)/count, modelPoints, niters );
            }
        }
    }

  if( maxGoodCount > 0 )
    {
      if( mask != mask0 )
	cvCopy( mask, mask0 );
      result = true;
    }

  return result;
}
*/

bool CvModelEstimator2::runMAPSAC( const CvMat* m1, const CvMat* m2, CvMat* model,
				   CvMat* mask, double reprojThreshold, double confidence, int maxIters )
{
  //const double outlierRatio = 0.45;
  bool result = false;
  cv::Ptr<CvMat> models;
  cv::Ptr<CvMat> ms1, ms2;
  cv::Ptr<CvMat> err;

  int iter, niters = maxIters;
  int count = m1->rows*m1->cols;
  double minSSE = DBL_MAX, sigma;

  CV_Assert( CV_ARE_SIZES_EQ(m1, m2) && CV_ARE_SIZES_EQ(m1, mask) );

  if( count < modelPoints )
    return false;

  models = cvCreateMat( modelSize.height*maxBasicSolutions, modelSize.width, CV_64FC1 );
  err = cvCreateMat( 1, count, CV_32FC1 );
  float* err_ptr = err->data.fl;

  if( count > modelPoints )
    {
      ms1 = cvCreateMat( 1, modelPoints, m1->type );
      ms2 = cvCreateMat( 1, modelPoints, m2->type );
    }
  else
    {
      niters = 1;
      ms1 = cvCloneMat(m1);
      ms2 = cvCloneMat(m2);
    }

  //niters = cvRound(log(1-confidence)/log(1-pow(1-outlierRatio,(double)modelPoints)));
  //niters = MIN( MAX(niters, 3), maxIters );

  for( iter = 0; iter < niters; iter++ )
    {
      int i, nmodels;
     
      if( count > modelPoints )
        {
	  bool found = getSubset( m1, m2, ms1, ms2, 300 );
	  if( !found )
            {
	      if( iter == 0 )
		return false;
	      break;
            }
        }

      nmodels = runKernel( ms1, ms2, models );
      if( nmodels <= 0 )
	continue;
      for( i = 0; i < nmodels; i++ )
        {
	  double SSE = 0;
	  CvMat model_i;
	  cvGetRows( models, &model_i, i*modelSize.height, (i+1)*modelSize.height );
	  computeReprojError( m1, m2, &model_i, err );
	  //icvSortDistances( err->data.i, count, 0 );

	  //double median = count % 2 != 0 ?
	  //   err->data.fl[count/2] : (err->data.fl[count/2-1] + err->data.fl[count/2])*0.5; 
	  for(int i2 = 0; i2 < count; i++)   //Get Sum Squared Error
	    {
	      if(err_ptr[i2] > reprojThreshold)
		SSE += (reprojThreshold*reprojThreshold);
	      else
		SSE += (err_ptr[i2]*err_ptr[i2]);
	    }

            if( SSE < minSSE )
	      {
                minSSE = SSE;
                cvCopy( &model_i, model );
	      }
        }
    }

  if( minSSE < DBL_MAX )
    {
      //sigma = 2.5*1.4826*(1 + 5./(count - modelPoints))*sqrt(minMedian);
      //sigma = MAX( sigma, 0.001 );

      count = findInliers( m1, m2, model, err, mask, reprojThreshold );
      result = count >= modelPoints;
    }

  return result;
}






int cv::TranslationEstimator::runKernel( const CvMat* m1, const CvMat* m2, CvMat* model )
{
    cout << "DEBUG:ENTER RUNKERNEL" << endl;
    const Point2d* from = reinterpret_cast<const Point2d*>(m1->data.ptr);

    cout << "ROWS = " << model->rows << endl;
    cout << "COLS = " << model->cols << endl;
    Mat modelMat = Mat::zeros(1,2,CV_64F);
 
    modelMat.at<double>(0,0) = from[0].x;
    modelMat.at<double>(0,1) = from[0].y;

    CvMat cvmat_model = modelMat;

    //    model = &cvmat_model;
    CvMat cvX;
    cvReshape(model, &cvX, 1, 2);
    double* cvXptr = reinterpret_cast<double*>(model->data.ptr);
    Point2d* modelptr = reinterpret_cast<Point2d*>(model->data.ptr);
    cvXptr[0] = from[0].x;
    cvXptr[1] = from[0].y;

    cout << "RunKernel Model = " << modelptr[0] << endl;
    cout << "ROWS = " << model->rows << endl;
    cout << "COLS = " << model->cols << endl;
    return 1;
}

void cv::TranslationEstimator::computeReprojError( const CvMat* m1, const CvMat* m2, const CvMat* model, CvMat* error )
{
  cout << "DEBUG:ENTER COMPUTEREPROJERROR" << endl;
    int count = m1->rows * m1->cols;
    const Point2d* from = reinterpret_cast<const Point2d*>(m1->data.ptr);
    const Point2d* to   = reinterpret_cast<const Point2d*>(m2->data.ptr);
    const double* F = model->data.db;
    float* err = error->data.fl;

    for(int i = 0; i < count; i++ )
    {
        const Point2d& f = from[i];
        //const Point2d& t = to[i];

        //double a = F[0]*f.x + F[1]*f.y + F[ 2]*f.z + F[ 3] - t.x;
        //double b = F[4]*f.x + F[5]*f.y + F[ 6]*f.z + F[ 7] - t.y;
        //double c = F[8]*f.x + F[9]*f.y + F[10]*f.z + F[11] - t.z;
	
	double xshift = f.x - F[0];
	double yshift = f.y - F[1];

        err[i] = (float)sqrt(xshift*xshift + yshift*yshift);

	cout << "Error " << err[i] << endl;
    }
    cout << "DEBUG LEAVING REPROJ ERR" << endl;
}

int estimateTranslation(InputArray _from, OutputArray _out, OutputArray _inliers,
                         double param1, double param2)
{
    Mat from = _from.getMat(), to = _from.getMat();
    int count = from.checkVector(2);

    CV_Assert( count >= 0 );

    _out.create(1, 2, CV_64F);
    Mat out = _out.getMat();

    Mat inliers(1, count, CV_8U);
    inliers = Scalar::all(1);

    Mat dFrom, dTo;
    from.convertTo(dFrom, CV_64F);
    to.convertTo(dTo, CV_64F);
    dFrom = dFrom.reshape(2, 1);
    dTo = dTo.reshape(2, 1);

    CvMat F2x1 = out;
    CvMat mask = inliers;
    CvMat m1 = dFrom;
    CvMat m2 = dTo;

    const double epsilon = numeric_limits<double>::epsilon();
    param1 = param1 <= 0 ? 3 : param1;
    param2 = (param2 < epsilon) ? 0.99 : (param2 > 1 - epsilon) ? 0.99 : param2;

    int ok = TranslationEstimator().runRANSAC(&m1, &m2, &F2x1, &mask, param1, param2,2000 );
      cout << ok << endl;
    if( _inliers.needed() )
        transpose(inliers, _inliers);

    return ok;
    }


int main(int argc, char** argv)
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
  
  int ok = estimateTranslation(points1,model,mask,3,0.9);

  cout << "MODEL: " << model << endl;
  cout << "MASK: " << mask << endl;
  cout << "POINTS1: " << points1 << endl;
  cout << "POINTS2: " << points2 << endl;
  cout << ok << endl;
}
