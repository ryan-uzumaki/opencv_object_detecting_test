#include <iostream>
#include <opencv2\opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include "opencv2/objdetect.hpp"
#include <sstream>

using namespace std;
using namespace cv;

class Process {
public:
	double get_distance(int W, int P);
	Point frame_1_center_point(Mat& image);
	string Convert(float Num);
	void object_recognition(Mat& image);
};
