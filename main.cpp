#include <algorithm>
#include <iostream>
#include <opencv2\opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include "opencv2/objdetect.hpp"
#include "stdlib.h"
#include "core/core.hpp"  
#include "highgui/highgui.hpp"  
#include "imgproc/imgproc.hpp"  
#include <cmath>
#include <sstream>


using namespace std;
using namespace cv;
int known_W = 9;
int known_P = 510;

double get_distance(int W, int P);
void object_recognition(Mat& image);
Point frame_1_center_point(Mat& image);
string Convert(float Num);


//template<class ForwardIterator>
//inline size_t argmin(ForwardIterator first, ForwardIterator last)
//{
//	return std::distance(first, std::min_element(first, last));
//}
//
//template<class ForwardIterator>
//inline size_t argmax(ForwardIterator first, ForwardIterator last)
//{
//	return std::distance(first, std::max_element(first, last));
//}


int main() {
	VideoCapture capture(0);
	VideoCapture capture_1(0);
	Mat frame;
	Mat frame_1;
	while (true) {
		capture.read(frame);
		waitKey(1);
		capture_1.read(frame_1);
		if (frame.empty()||frame_1.empty()) {
			break;
		}
		Mat temp = Mat::zeros(frame.size(), frame.type());
		Mat m = Mat::zeros(frame.size(), frame.type());
		addWeighted(frame, 0.19, m, 0.0, 0, temp);
		Mat dst;
		bilateralFilter(temp, dst, 5, 20, 20);
		Mat m_ResImg;
		cvtColor(dst, m_ResImg, COLOR_BGR2HSV);
		Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
		erode(m_ResImg, m_ResImg, element);//进行腐蚀操作
		erode(m_ResImg, m_ResImg, element);//进行腐蚀操作
		erode(m_ResImg, m_ResImg, element);//进行腐蚀操作
		Mat mask;
		inRange(m_ResImg, Scalar(100, 43, 46), Scalar(124, 255, 255), mask);
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		findContours(mask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());
		vector<double> area;
		for (int i = 0; i < contours.size(); i++) {
			area.push_back(contourArea(contours[i]));
		}
		size_t maxIndex = argmax(area.begin(), area.end());
		Rect ret_1 = boundingRect(contours[maxIndex]);
		int avgX, avgY;
		avgX = (ret_1.x + ret_1.width) / 2;//x-axis middle point
		avgY = (ret_1.y + ret_1.height) / 2;//y-axis middle point
		for (int i = 0; i < contours.size(); i++) {
			//for (int j = 0; j < contours[i].size(); j++) {
			//	Point P = Point(contours[i][j].x, contours[i][j].y);
			//	Mat Contours = Mat::zeros(m_ResImg.size(), CV_8UC1);  //绘制
			//	Contours.at<uchar>(P) = 255;
			//}
			Rect box(ret_1.x, ret_1.y, ret_1.width, ret_1.height);
			rectangle(frame, box, Scalar(0, 0, 255), 2, 8, 0);
			drawContours(frame, contours, maxIndex, Scalar(0, 255, 0), 2, 8, hierarchy);
		}
		double dist = get_distance(known_W, ret_1.width);
		string dist_str = Convert(dist);
		putText(frame, "Distance:" + dist_str + "cm", Point(50, 50), FONT_HERSHEY_COMPLEX, 1, Scalar(50, 250, 50), 2, 8);
		namedWindow("detected", WINDOW_FREERATIO);
		imshow("detected", frame);
		int c = waitKey(1);
		if (c == 27) { // 退出
			break;
		}
	}
	//capture.release();
	return 0;
}







void detect_object(Mat& imageSource) {
	//imshow("Source Image", imageSource);
	Mat image = Mat::zeros(imageSource.size(), imageSource.type());
	image = imageSource.clone();
	//GaussianBlur(imageSource, image, Size(3, 3), 0);
	Canny(image, image, 50, 100);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(image, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());
	Mat imageContours = Mat::zeros(image.size(), CV_8UC1);
	Mat Contours = Mat::zeros(image.size(), CV_8UC1);  //绘制
	for (int i = 0; i < contours.size(); i++) {
		//contours[i]代表的是第i个轮廓，contours[i].size()代表的是第i个轮廓上所有的像素点数
		for (int j = 0; j < contours[i].size(); j++) {
			//绘制出contours向量内所有的像素点
			Point P = Point(contours[i][j].x, contours[i][j].y);
			Contours.at<uchar>(P) = 255;
		}

		//输出hierarchy向量内容
		/*char ch[256];
		sprintf_s(ch, "%d", i);
		string str = ch;
		cout << "向量hierarchy的第" << str << " 个元素内容为：" << endl << hierarchy[i] << endl << endl;*/

		//绘制轮廓
		drawContours(imageContours, contours, i, Scalar(255), 1, 8, hierarchy);
	}
	imshow("Contours Image", imageContours); //轮廓
	//imshow("Point of Contours", Contours);   //向量contours内保存的所有轮廓点集
	waitKey(0);
}

