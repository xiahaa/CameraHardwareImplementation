#ifndef __REFINED_CIRCULAR_POSITIONING_SYSTEM_H__
#define __REFINED_CIRCULAR_POSITIONING_SYSTEM_H__

#include "opencv2/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"

struct __ellipseFeatures_t {
	float x;
	float y;//ellipse center
	float a;//major axis
	float b;//minor axis
	float theta;//
	float area;
	float v0;
	float v1;
	int bbwith;
	int bbheight;

	float A, B, C, D, E, F;

	__ellipseFeatures_t(void):x(0), y(0), a(0), b(0), theta(0), v0(0), v1(0),
		A(0), B(0), C(0), D(0), E(0), F(0), bbwith(0), bbheight(0){}

	__ellipseFeatures_t& operator=(const __ellipseFeatures_t &other)
	{
		if (&other == this)
			return *this;
		memcpy(this, &other, sizeof(__ellipseFeatures_t));
		return *this;
	}
};

class ringCircularPattern {
public:
	ringCircularPattern() : valid(false){}

	ringCircularPattern(const std::pair<int, int> &_pair, __ellipseFeatures_t &_in, const __ellipseFeatures_t &_out) {
		matchpair.first = _pair.first;
		matchpair.second = _pair.second;
		r33 = cv::Mat::eye(3, 3, CV_32F);
		cv::Rodrigues(r33, r3);
		t(0) = t(1) = t(2) = 0;
		setEllipse(_in, _out);
		valid = true;
	}

	~ringCircularPattern() {}

	bool setEllipse(const __ellipseFeatures_t &_in, const __ellipseFeatures_t &_out)
	{
		inner = _in;
		outter = _out;
		return true;
	}

public:
	bool valid;
	cv::Vec3f t;
	cv::Mat r33;
	cv::Mat r3;
	std::pair<int, int>  matchpair;
	__ellipseFeatures_t inner;
	__ellipseFeatures_t outter;
};

class circularPatternBasedLocSystems {
public:
	circularPatternBasedLocSystems(const cv::Mat &K, const cv::Mat &dist, float _innerdiameter, float _outterdiameter, float _xaxisl, float _yaxisl):
		innerdiameter(_innerdiameter), outterdiameter(_outterdiameter), xaxisl(_xaxisl), yaxisl(_yaxisl)
	{
		K.copyTo(camK);
		dist.copyTo(distCoeff);
		ringCircles.clear();
		coordinateTransform = cv::Matx33f(1, 0, 0, 0, 1, 0, 0, 0, 1);
		setAxis = false;
	}
	~circularPatternBasedLocSystems() {}

	int detectPatterns(const cv::Mat &frame_gray, bool do_tracking);
	
	void drawPatterns(cv::Mat frame_rgb);

	void localization();

	bool setAxisFrame(std::vector<cv::Point> &click, std::string &axisFile);

	void read_axis(const std::string& file);

	void draw_axis(cv::Mat& image);
protected:
	void undistort(float x_in, float y_in, float& x_out, float& y_out);
	void getpos(const ringCircularPattern &circle, cv::Vec3f &position, cv::Vec3f rotation);
public:
	bool setAxis;
	float innerdiameter;
	float outterdiameter;
	float xaxisl;
	float yaxisl;
	cv::Mat camK;
	cv::Mat distCoeff;
	cv::Matx33f coordinateTransform;// from camera coordiante to user-defined coordinate
	std::vector<ringCircularPattern> ringCircles;
	cv::Point frame[4];
};

#endif
