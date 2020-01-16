//#include "_modelest.h"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"

using namespace cv;

int estimateTranslation(InputArray _from, OutputArray _out, OutputArray _inliers, double param1, double param2);

