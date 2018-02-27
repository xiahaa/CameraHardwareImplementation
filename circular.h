/*
* Date:      2010
* Author:   Tom Krajnik, Matias Nitsche
*/

#ifndef __CIRCLE_DETECTOR_H__
#define __CIRCLE_DETECTOR_H__

#include <opencv2/opencv.hpp>
#include <math.h>
#include <vector>

#define WHYCON_DEFAULT_OUTER_DIAMETER 0.114//0.122
#define WHYCON_DEFAULT_INNER_DIAMETER 0.047//0.050
#define WHYCON_DEFAULT_DIAMETER_RATIO (WHYCON_DEFAULT_INNER_DIAMETER/WHYCON_DEFAULT_OUTER_DIAMETER)

namespace cv {
	class CircleDetector
	{
	public:
		class Circle;
		class Context;

		CircleDetector() {};
		CircleDetector(int width, int height, Context* context, float diameter_ratio = WHYCON_DEFAULT_DIAMETER_RATIO);
		~CircleDetector();

		Circle detect(const cv::Mat& image, const Circle& previous_circle = cv::CircleDetector::Circle());
		bool examineCircle(const cv::Mat& image, Circle& circle, int ii, float areaRatio);
		void cover_last_detected(cv::Mat& image);

		void improveEllipse(const cv::Mat& image, Circle& c);
		int get_threshold(void) const;

	private:

		int minSize, maxSize;
		float diameterRatio;
		int thresholdStep;
		float circularTolerance;
		float ratioTolerance;
		float centerDistanceToleranceRatio;
		int centerDistanceToleranceAbs;

		float outerAreaRatio, innerAreaRatio, areasRatio;
		int width, height, len, siz;

		int threshold, threshold_counter;
		void change_threshold(void);

		int queueStart, queueEnd, queueOldStart, numSegments;

		Context* context;

	public:
		class Circle {
		public:
			Circle(void);

			float x;
			float y;
			int size;
			int maxy, maxx, miny, minx;
			int mean;
			int type;
			float roundness;
			float bwRatio;
			bool round;
			bool valid;
			float m0, m1; // axis dimensions
			float v0, v1; // axis (v0,v1) and (v1,-v0)

			void write(cv::FileStorage& fs) const;
			void read(const cv::FileNode& node);
			cv::Point2i cv::CircleDetector::Circle::returnEllipseCenter() const;
			void draw(cv::Mat& image, const std::string& text = std::string(), cv::Vec3b color = cv::Vec3b(0, 255, 0), float thickness = 1) const;
		};

		class Context {
		public:
			Context() {}
			Context(int _width, int _height);

			bool setParams(int _width, int _height)
			{
				height = _height;
				width = _width;
				return true;
			}

			void debug_buffer(const cv::Mat& image, cv::Mat& img);

			std::vector<int> buffer, queue;
			int width, height;

		private:
			void cleanup(const Circle& c, bool fast_cleanup);
			friend class CircleDetector;
		};
	};
}

namespace cv {
	class ManyCircleDetector {
	public:
		ManyCircleDetector() {}

		bool setParams(int _number_of_circles, int _width, int _height, float _diameter_ratio = WHYCON_DEFAULT_DIAMETER_RATIO)
		{
			context.setParams(_width, _height);

			width = _width;
			height = _height;
			number_of_circles = _number_of_circles;

			circles.resize(number_of_circles);
			detectors.resize(number_of_circles, CircleDetector(width, height, &context, _diameter_ratio));
			return true;
		}

		ManyCircleDetector(int number_of_circles, int width, int height,
			float diameter_ratio = WHYCON_DEFAULT_DIAMETER_RATIO);
		~ManyCircleDetector(void);

		bool detect(const cv::Mat& image, cv::Mat& cimg, bool reset = true, int max_attempts = 1, int refine_max_step = 1);

		std::vector<CircleDetector::Circle> circles;

		CircleDetector::Context context;

	private:
		int width, height, number_of_circles;
		std::vector<CircleDetector> detectors;
	};
}

namespace cv {
	class LocalizationSystem {
	public:
		LocalizationSystem() {}
		~LocalizationSystem() {}

		bool setParams(int _targets, int _width, int _height, const cv::Mat& _K, const cv::Mat& _dist_coeff,
			float _outer_diameter = WHYCON_DEFAULT_OUTER_DIAMETER, float _inner_diameter = WHYCON_DEFAULT_INNER_DIAMETER)
		{
			xscale = 1;
			yscale = 1;
			targets = _targets;
			width = _width;
			height = _height;
			axis_set = false;
			circle_diameter = _outer_diameter;


			detector.setParams(_targets, _width, _height, _inner_diameter / _outer_diameter);

			_K.copyTo(K);
			_dist_coeff.copyTo(dist_coeff);

			fc[0] = K.at<double>(0, 0);
			fc[1] = K.at<double>(1, 1);
			cc[0] = K.at<double>(0, 2);
			cc[1] = K.at<double>(1, 2);

			std::cout.precision(30);
			std::cout << "fc " << fc[0] << " " << fc[1] << std::endl;
			std::cout << "cc " << cc[0] << " " << cc[1] << std::endl;
			kc[0] = 1;
			std::cout << "kc " << kc[0] << " ";
			for (int i = 0; i < 5; i++) {
				kc[i + 1] = dist_coeff.at<double>(i);
				std::cout << kc[i + 1] << " ";
			}
			std::cout << std::endl;

			coordinates_transform = cv::Matx33f(1, 0, 0, 0, 1, 0, 0, 0, 1);

			precompute_undistort_map();

			return true;
		}

		LocalizationSystem(int targets, int width, int height, const cv::Mat& K, const cv::Mat& dist_coeff,
			float outer_diameter = WHYCON_DEFAULT_OUTER_DIAMETER, float inner_diameter = WHYCON_DEFAULT_INNER_DIAMETER);

		bool set_axis(const cv::Mat& image, int attempts = 1, int max_refine = 1, const std::string& output = std::string());
		void read_axis(const std::string& input);
		void draw_axis(cv::Mat& image);

		bool localize(const cv::Mat& image, bool reset = false, int attempts = 1, int max_refine = 1);

		float xscale, yscale;

		// TODO: use double?
		struct Pose {
			cv::Vec3f pos;
			cv::Vec3f rot; // pitch, roll, yaw
		};

		Pose get_pose(int id) const;
		Pose get_pose(const CircleDetector::Circle& circle) const;
		const CircleDetector::Circle& get_circle(int id);

		Pose get_transformed_pose(int id) const;
		Pose get_transformed_pose(const CircleDetector::Circle& circle) const;

		static void load_matlab_calibration(const std::string& calib_file, cv::Mat& K, cv::Mat& dist_coeff);
		static void load_opencv_calibration(const std::string& calib_file, cv::Mat& K, cv::Mat& dist_coeff);

		CircleDetector::Circle origin_circles[4]; // center, X, Y

		cv::Matx33f coordinates_transform;
		ManyCircleDetector detector;

		int targets, width, height;
		bool axis_set;

	private:
		cv::Mat K, dist_coeff;
		float circle_diameter;
		double fc[2]; // focal length X,Y
		double cc[2]; // principal point X,Y
		double kc[6]; // distortion coefficients
		void transform(float x_in, float y_in, float& x_out, float& y_out) const;

		void precompute_undistort_map(void);
		cv::Mat undistort_map;


		cv::Vec3f eigen(double data[]) const;
	};
}

#endif
