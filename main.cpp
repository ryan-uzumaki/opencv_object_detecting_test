#include <algorithm>
#include <iostream>
#include <opencv2\opencv.hpp>
//#include <opencv2/imgproc/types_c.h>
//#include "opencv2/objdetect.hpp"
//#include "stdlib.h"
//#include "core/core.hpp"  
//#include "highgui/highgui.hpp"  
//#include "imgproc/imgproc.hpp"  
#include <cmath>
#include <sstream>
#include <iterator>
#include "Process.hpp"
#include <opencv2\video\video.hpp>


using namespace std;
using namespace cv;

//class Process {
//public:
//	double get_distance(int W, int P);
//	void predict(Mat& image, Mat& image_temp, int avgX_temp, int avgY_temp);
//	string Convert(float Num);
//	void object_recognition(Mat& image, Mat& image_temp_1);
//};
////
////template<class ForwardIterator>
////inline size_t argmin(ForwardIterator first, ForwardIterator last)
////{
////	return std::distance(first, std::min_element(first, last));
////}
////
////template<class ForwardIterator>
////inline size_t argmax(ForwardIterator first, ForwardIterator last)
////{
////	return std::distance(first, std::max_element(first, last));
////}
//
//string Process::Convert(float Num)
//{
//	std::ostringstream oss;
//	oss << Num;
//	std::string str(oss.str());
//	return str;
//}
//
//
//double Process::get_distance(int W, int P) {
//	double F = 550;
//	double D = 0;
//	D = (W * F) / P;
//	return D;
//}
//
//
//void Process::object_recognition(Mat& image,Mat& image_temp_1) {
//	Process pr;
//	int known_W = 9;
//	int known_P = 510;
//	Mat temp = Mat::zeros(image.size(), image.type());
//	Mat m = Mat::zeros(image.size(), image.type());
//	addWeighted(image, 0.19, m, 0.0, 0, temp);
//	Mat dst;
//	bilateralFilter(temp, dst, 5, 20, 20);
//	Mat m_ResImg;
//	cvtColor(dst, m_ResImg, COLOR_BGR2HSV);
//	Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
//	erode(m_ResImg, m_ResImg, element);//进行腐蚀操作
//	erode(m_ResImg, m_ResImg, element);//进行腐蚀操作
//	erode(m_ResImg, m_ResImg, element);//进行腐蚀操作
//	Mat mask;
//	inRange(m_ResImg, Scalar(100, 43, 46), Scalar(124, 255, 255), mask);
//	vector<vector<Point>> contours;
//	vector<Vec4i> hierarchy;
//	findContours(mask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());
//	vector<double> area;
//	for (int i = 0; i < contours.size(); i++) {
//		area.push_back(contourArea(contours[i]));
//	}
//	size_t maxIndex = distance(area.begin(), max_element(area.begin(),area.end()));
//	Rect ret_1 = boundingRect(contours[maxIndex]);
//	int avgX, avgY;
//	avgX = ret_1.x;
//	avgY = ret_1.y;
//	for (int i = 0; i < contours.size(); i++) {
//		//for (int j = 0; j < contours[i].size(); j++) {
//		//	Point P = Point(contours[i][j].x, contours[i][j].y);
//		//	Mat Contours = Mat::zeros(m_ResImg.size(), CV_8UC1);  //绘制
//		//	Contours.at<uchar>(P) = 255;
//		//}
//		Rect box(ret_1.x, ret_1.y, ret_1.width, ret_1.height);
//		rectangle(image, box, Scalar(0, 0, 255), 2, 8, 0);
//		drawContours(image, contours, maxIndex, Scalar(0, 255, 0), 2, 8, hierarchy);
//	}
//	pr.predict(image, image_temp_1, avgX, avgY);
//	double dist = pr.get_distance(known_W, ret_1.width);
//	string dist_str = pr.Convert(dist);
//	putText(image, "Distance:" + dist_str + "cm", Point(50, 50), FONT_HERSHEY_COMPLEX, 1, Scalar(50, 250, 50), 2, 8);
//	namedWindow("detected", WINDOW_FREERATIO);
//	imshow("detected", image);
//}
//
//void Process::predict(Mat& image, Mat& image_temp, int avgX_temp, int avgY_temp) {
//	Mat frame_temp;
//	frame_temp = image_temp.clone();
//	Process pr;
//	int known_W = 9;
//	int known_P = 510;
//	Mat temp = Mat::zeros(image_temp.size(), image_temp.type());
//	Mat m = Mat::zeros(image_temp.size(), image_temp.type());
//	addWeighted(image_temp, 0.19, m, 0.0, 0, temp);
//	Mat dst;
//	bilateralFilter(temp, dst, 5, 20, 20);
//	Mat m_ResImg;
//	cvtColor(dst, m_ResImg, COLOR_BGR2HSV);
//	Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
//	erode(m_ResImg, m_ResImg, element);//进行腐蚀操作
//	erode(m_ResImg, m_ResImg, element);//进行腐蚀操作
//	erode(m_ResImg, m_ResImg, element);//进行腐蚀操作
//	Mat mask;
//	inRange(m_ResImg, Scalar(100, 43, 46), Scalar(124, 255, 255), mask);
//	vector<vector<Point>> contours;
//	vector<Vec4i> hierarchy;
//	findContours(mask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());
//	vector<double> area;
//	for (int i = 0; i < contours.size(); i++) {
//		area.push_back(contourArea(contours[i]));
//	}
//	size_t maxIndex = distance(area.begin(), max_element(area.begin(), area.end()));
//	Rect ret_1 = boundingRect(contours[maxIndex]);
//	int avgX, avgY;
//	int avgX_mean, avgY_mean;
//	avgX = ret_1.x;
//	avgY = ret_1.y;
//	avgX_mean = round((0.3*avgX + 0.7*avgX_temp));
//	avgY_mean = round((0.3*avgY + 0.7*avgY_temp));
//	Rect ret_2(avgX_mean, avgY_mean, ret_1.width, ret_1.height);
//	Rect box(ret_2.x, ret_2.y, ret_2.width, ret_2.height);
//	rectangle(image, box, Scalar(255, 0, 0), 2, 8, 0);
//	/*for (int i = 0; i < contours.size(); i++) {
//		drawContours(image, contours, maxIndex, Scalar(0, 255, 0), 2, 8, hierarchy);
//	}*/
//}

//main entrance
//int main() {
//	VideoCapture capture(0);
//	VideoCapture capture_1(0);
//	Mat frame;
//	Mat frame_1;
//	Process object;
//	while (true) {
//		capture.read(frame);
//		waitKey(1);
//		capture_1.read(frame_1);
//		if (frame.empty()||frame_1.empty()) {
//			break;
//		}
//		object.object_recognition(frame,frame_1);
//		int c = waitKey(1);
//		if (c == 27) { // 退出
//			break;
//		}
//	}
//	//capture.release();
//	return 0;
//}

int main() {
	VideoCapture capture(0);
	capture.set(CAP_PROP_FRAME_WIDTH, 640);
	capture.set(CAP_PROP_FRAME_HEIGHT, 480);
	Process object;
	while (true) {
		Mat frame;
		Mat frame_1;
		//cout << capture.get(CAP_PROP_POS_MSEC) << endl;
		/*capture.read(frame);
		waitKey(1);
		capture_1.read(frame_1);*/
		capture >> frame;
		capture >> frame_1;
		if (frame.empty() || frame_1.empty()) {
			break;
		}
		object.object_recognition(frame,frame_1);
		/*namedWindow("capture", WINDOW_AUTOSIZE);
		imshow("capture", frame);
		imshow("capture", frame_1);*/
		//int c=waitKey(1000 / capture.get(CAP_PROP_FPS));
		int c = waitKey(10);
		if (c == 27) { // 退出
			break;
		}
	}
	capture.release();
	return 0;
}
