#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "precomp.hpp"

class TranslationEstimatorCallback : public PointSetRegistrator::Callback
{
public:
  int runKernel( InputArray _m1, InputArray _m2, OutputArray _model ) const;
  void computeError( InputArray _m1, InputArray _m2, InputArray _model, OutputArray _err ) const;
  bool checkSubset( InputArray _ms1, InputArray _ms2, int count ) const;
};

int estimateTranslationNew(InputArray _from, InputArray _to,
			     OutputArray _out, OutputArray _inliers,
			     double param1, double param2);

void testEstimateTranslationNew();
