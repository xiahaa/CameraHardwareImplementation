#include "refinedCircularPositioningSystem.h"
#include <iostream>
#include <fstream>

#include <iomanip>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <chrono>
#include "opencv2\core.hpp"
#include "opencv2/highgui.hpp"
#include "fitEllipse.h"
//#include "tbb\tbb.h"
#include <omp.h>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

using namespace std;

#pragma warning(disable : 4244)  

#define ESTIMATE_ELLIPSE_BY_MOMENTS	0
#define PARALLEL_COMPUTING			0

#ifndef MAX(a,b)	
#define MAX(a,b)	((a)>(b)?(a):(b))
#endif

#ifndef MIN(a,b)	
#define MIN(a,b)	((a)<(b)?(a):(b))
#endif

bool comp(const ringCircularPattern& lhs, const ringCircularPattern& rhs)
{
	return (lhs.outter.x*lhs.outter.y) < (rhs.outter.x*rhs.outter.y);
}

int todotracking()
{
	// step should be: threshold, crop, check and filter, 
	
	// do roundness check at coarse layer
	//std::vector<cv::Rect> bbois;
	//for (size_t i = 0; i < contours.size(); i++)
	//{
	//	if (contours[i].size() > minContourSize && contours[i].size() < maxContourSize)
	//	{
	//		double perimeter = cv::arcLength(contours[i], true);
	//		double area = cv::contourArea(contours[i], false);
	//		double roundness = 4 * CV_PI*area / perimeter / perimeter;
	//		double errRoundness = fabs(roundness - 1);
	//		if (errRoundness < roundnessTolerance)
	//		{
	//			cv::Rect bb = cv::boundingRect(contours[i]);
	//			//TODO, here, filter contours to close to border
	//			if (bb.x + bb.width + 20 < frame_gray.cols)
	//				bb.width += 20;
	//			if (bb.y + bb.height + 20 < frame_gray.rows)
	//				bb.height += 20;
	//			bbois.push_back(bb);
	//		}
	//	}
	//}

	/*minContourSize = minContourSize * resizeScale_inv;
	maxContourSize = maxContourSize * resizeScale_inv;

	std::vector<std::vector<cv::Point>> contoursoi;
	for (size_t i = 0; i < bbois.size(); i++)
	{
		float cx = bbois[i].x * resizeScale_inv - 10;
		float cy = bbois[i].y * resizeScale_inv - 10;
		float w = bbois[i].width * resizeScale_inv - 10;
		float h = bbois[i].height * resizeScale_inv - 10;

		cv::Mat srcoi = frame_gray(cv::Rect(cx, cy, w, h));
		cv::Mat binoi;
		cv::adaptiveThreshold(srcoi, binoi, 255, cv::ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 3, 0);

		std::vector<std::vector<cv::Point>> crs;
		std::vector<cv::Vec4i> hier;
		cv::findContours(binoi, crs, hier, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

		for (size_t i = 0; i < crs.size(); i++)
		{
			if (crs[i].size() > minContourSize && crs[i].size() < maxContourSize)
			{
				double perimeter = cv::arcLength(crs[i], true);
				double area = cv::contourArea(crs[i], false);
				double roundness = 4 * CV_PI*area / perimeter / perimeter;
				double errRoundness = fabs(roundness - 1);
				if (errRoundness < roundnessTolerance)
				{
					std::vector<cv::Point> c(crs[i].size());
					for (size_t j = 0; j < crs[i].size(); j++)
					{
						c[j].x = crs[i][j].x + cx;
						c[j].y = crs[i][j].y + cx;
					}
					contoursoi.push_back(c);
				}
			}
		}
	}*/
	return true;
}

class ParallelProcessing : public cv::ParallelLoopBody
{
public:
	ParallelProcessing(std::vector<std::vector<cv::Point> > &_contours, std::vector<uint8_t> _valid,
		float _resizeScale_inv, int _minContourSize, int _maxContourSize, double _roundnessTolerance)
		: contours(_contours), valid(_valid), resizeScale_inv(_resizeScale_inv), minContourSize(_minContourSize),
		maxContourSize(_maxContourSize), roundnessTolerance(_roundnessTolerance)
	{
	}

	virtual void operator()(const cv::Range &range) const
	{
		for (int r = range.start; r < range.end; r++)
		{
			//printf("%d %d\n", r, range.end);
			if ((int)(contours[r].size()) > minContourSize && (int)(contours[r].size()) < maxContourSize)
			{
				double perimeter = cv::arcLength(contours[r], true);
				double area = cv::contourArea(contours[r], false);
				double roundness = 4 * CV_PI*area / perimeter / perimeter;
				double errRoundness = fabs(roundness - 1);
				if (errRoundness < roundnessTolerance)
				{
					cv::Moments mm = cv::moments(contours[r]);
					//compute ellipse parameters
					__ellipseFeatures_t e;

#if ESTIMATE_ELLIPSE_BY_MOMENTS
					/* opencv, normalized moments exist some bugs */
					float scaleNomr = 1 / mm.m00;
					float nu11 = mm.mu11 * scaleNomr;
					float nu20 = mm.mu20 * scaleNomr;
					float nu02 = mm.mu02 * scaleNomr;

					float trace = nu20 + nu02;
					float det = 4 * nu11*nu11 + (nu20 - nu02) * (nu20 - nu02);
					if (det > 0) det = sqrt(det); else det = 0;//yes, that can happen as well:(
					float f0 = (trace + det) / 2;
					float f1 = (trace - det) / 2;

					/* cnetroid */
					e.x = mm.m10 / mm.m00 * resizeScale_inv;
					e.y = mm.m01 / mm.m00 * resizeScale_inv;
					/* major and minor axis */
					e.a = 2 * sqrt(f0) * resizeScale_inv;
					e.b = 2 * sqrt(f1) * resizeScale_inv;

					float dem = nu11*nu11 + (nu20 - f0)*(nu20 - f0);
					float sdem = sqrt(dem);

					if (nu11 != 0) {                               //aligned ?
						e.v0 = -nu11 / sdem; // no-> standard calculatiion
						e.v1 = (nu20 - f0) / sdem;
					}
					else {
						e.v0 = e.v1 = 0;
						if (nu20 > nu02) e.v0 = 1.0; else e.v1 = 1.0;   // aligned, hm, is is aligned with x or y ?
					}
					e.theta = 0.5 * atan2(2 * nu11, nu20 - nu02);
					e.area = area * resizeScale_inv * resizeScale_inv;
#else /* direct ellipse fitting */

#endif
					float circularity = CV_PI * (e.a)*(e.b) / e.area;
					if (fabsf(circularity - 1) < 0.3) {
						//can1.push_back(r);
						//canMoments.push_back(e);
						valid[r] = 1;
					}
					else valid[r] = 0;
				} else valid[r] = 0;
			} else valid[r] = 0;
		}
	}

	ParallelProcessing& operator=(const ParallelProcessing &) {
		return *this;
	};

private:
	float resizeScale_inv;
	int minContourSize;
	int maxContourSize;
	double roundnessTolerance;

	std::vector<std::vector<cv::Point> > &contours;
	std::vector<uint8_t> &valid;
};


/*
* extract connected components from given image.
*/
bool componentConnect(const cv::Mat &src, std::vector<int> &label,
	std::vector<std::vector<cv::Point2i> > &components, char *buf, char *rtype,
	int invalidVal,
	bool filterSmall, float smallRegionSize,
	bool filterBig, float bigRegionSize)
{
	//pixel label
	cv::Mat labelimg = cv::Mat::zeros(src.size(), CV_32SC1);
	// region type
	memset(rtype, 0, sizeof(char)*src.cols*src.rows);

	cv::Point2i *ptrBuf = (cv::Point2i *)buf;

	const int width = src.cols, height = src.rows;
	const int smallResionType = 1;
	const int smallComponentThresh = int(smallRegionSize * height * width);
	const int bigResionType = 2;
	const int bigComponentThresh = filterBig ?
		int(bigRegionSize * height * width) : (height * width);

	int currlabel = 1;
	const int unlabeled = 0;

	for (int i = 0; i < src.rows; i++)
	{
		const uchar *ptr = src.ptr<uchar>(i);
		int* lptr = labelimg.ptr<int>(i);
		for (int j = 0; j < src.cols; j++)
		{
			if (ptr[j] == invalidVal) continue;
			if (lptr[j] != unlabeled)
			{
				if (rtype[lptr[j]] == smallResionType)
				{
				}
				continue;
			}

			// valid
			lptr[j] = currlabel;
			cv::Point2i *base = ptrBuf;
			*ptrBuf = cv::Point2i(j, i); ptrBuf++;
			int componentSize = 1;
			std::vector<cv::Point2i> candidates;
			while (ptrBuf > base)
			{
				ptrBuf--;
				cv::Point2i pt = *(ptrBuf);
				candidates.push_back(pt);
				const uchar *rptr = src.ptr<uchar>(pt.y);
				int* rlptr = labelimg.ptr<int>(pt.y);

				//left
				if ((pt.x - 1) >= 0 && rptr[pt.x - 1] != invalidVal
					&& rlptr[pt.x - 1] == unlabeled)
				{
					componentSize++;
					rlptr[pt.x - 1] = currlabel;
					*ptrBuf++ = cv::Point2i(pt.x - 1, pt.y);
				}
				//right
				if ((pt.x + 1) < src.cols && rptr[pt.x + 1] != invalidVal
					&& rlptr[pt.x + 1] == unlabeled)
				{
					componentSize++;
					rlptr[pt.x + 1] = currlabel;
					*ptrBuf++ = cv::Point2i(pt.x + 1, pt.y);
				}
				//up
				if ((pt.y - 1) >= 0 && rptr[pt.x - width] != invalidVal
					&& rlptr[pt.x - width] == unlabeled)
				{
					componentSize++;
					rlptr[pt.x - width] = currlabel;
					*ptrBuf++ = cv::Point2i(pt.x, pt.y - 1);
				}
				//down
				if ((pt.y + 1) < src.rows && rptr[pt.x + width] != invalidVal
					&& rlptr[pt.x + width] == unlabeled)
				{
					componentSize++;
					rlptr[pt.x + width] = currlabel;
					*ptrBuf++ = cv::Point2i(pt.x, pt.y + 1);
				}
			}

			// end
			if (componentSize < smallComponentThresh)
			{
				// small region
				rtype[currlabel] = 1;
			}
			else if (componentSize > bigComponentThresh)
			{
				// big region
				rtype[currlabel] = 2;
			}
			else
			{
				// suitable
				label.push_back(currlabel);
				components.push_back(candidates);
			}
			currlabel++;
		}
	}

	return true;
}


bool extractContour(
	const cv::Mat &binary,
	const std::vector < std::vector<cv::Point> > &components,
	std::vector<std::vector<cv::Point>> &contours)
{
	contours.resize(components.size());
	const int width = binary.cols;
	const int height = binary.rows;

	for (size_t i = 0; i < components.size(); i++)
	{
		for (size_t j = 0; j < components[i].size(); j++)
		{
			//check 4 neighbors
			auto pt = components[i][j];
			const uchar *ptr = &(binary.ptr<uchar>(pt.y)[pt.x]);
			if ((pt.x - 1) >= 0 && ptr[-1] == 0) {
				contours[i].push_back(pt);
				continue;
			}
			if ((pt.x + 1) < width && ptr[+1] == 0) {
				contours[i].push_back(pt);
				continue;
			}
			if ((pt.y - 1) >= 0 && ptr[-width] == 0) {
				contours[i].push_back(pt);
				continue;
			}
			if ((pt.y + 1) < height && ptr[+width] == 0) {
				contours[i].push_back(pt);
				continue;
			}
		}
	}

	return true;
}

int circularPatternBasedLocSystems::detectPatterns(const cv::Mat &frame_gray, bool do_tracking)
{
	const int srcwidth = frame_gray.cols;
	const int srcheight = frame_gray.rows;
	// do we really need full resolution
	double resizeScale = 1;
	double resizeScale_inv = 1;
	// split image
	std::vector<std::vector<cv::Point> > contours;

	std::chrono::high_resolution_clock::time_point t1, t2, t3;
	if (!do_tracking)
	{
		t1 = std::chrono::high_resolution_clock::now();

		cv::Mat frame_gray_small;
		//
		cv::resize(frame_gray, frame_gray_small, cv::Size(0, 0), resizeScale, resizeScale, CV_INTER_NN);

		// threshold
		cv::Mat binary;
		cv::adaptiveThreshold(frame_gray_small, binary, 255, cv::ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 7, 0);
		//cv::threshold(frame_gray, binary, 25, 255, CV_THRESH_BINARY_INV);

#if 1	// USE_ERODE
	// image erode and dialate
		{
			int erosion_type = cv::MORPH_RECT;
			int winSize = 3;
			cv::Mat element = cv::getStructuringElement(erosion_type, cv::Size(winSize, winSize),
				cv::Point(-1, -1));
			/// Apply the erosion operation
			erode(binary, binary, element);
			int dilation_type = cv::MORPH_RECT;
			dilate(binary, binary, element);
		}
#endif

		t2 = std::chrono::high_resolution_clock::now();
		std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << std::endl;

		std::vector<cv::Vec4i> hierarchy;
		//std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
		cv::findContours(binary, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
		t3 = std::chrono::high_resolution_clock::now();
		//cv::Mat binaryc;
		//cv::cvtColor(binary, binaryc, CV_GRAY2BGR);
		//cv::namedWindow("binary", CV_WINDOW_NORMAL);
		//cv::imshow("binary", binaryc);
		//cv::waitKey(10);

		//cv::Mat components = cv::Mat::zeros(binary.size(), CV_8UC1);
		//std::vector<std::vector<cv::Point2i> > componentPts;
		//std::vector<int> label;
		//char *buf = (char *)malloc(sizeof(cv::Point2i)*binary.cols*binary.rows);
		//char *rtype = (char *)malloc(sizeof(char)*binary.cols*binary.rows);
		//std::vector<std::vector<cv::Point> > contours1;
		//std::chrono::high_resolution_clock::time_point t4 = std::chrono::high_resolution_clock::now();
		//componentConnect(binary, label, componentPts, buf, rtype, 0, true, 0.001, true, 0.01);
		//extractContour(binary, componentPts, contours1);	
		//std::chrono::high_resolution_clock::time_point t5 = std::chrono::high_resolution_clock::now();
		std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count() << std::endl;
		//std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t4).count() << std::endl;
	}
	else
	{
		std::vector<std::vector<std::vector<cv::Point> > > contours_openmp;
		t1 = std::chrono::high_resolution_clock::now();
		int num_of_threads = omp_get_max_threads();
		contours_openmp.resize(num_of_threads);
#if PARALLEL_COMPUTING
#pragma omp parallel for
#endif
		for (int i = 0; i < ringCircles.size(); i++)
		{
			int id = omp_get_thread_num();
			cv::Rect bb;
			bb.x = MAX(ringCircles[i].outter.x - ringCircles[i].outter.bbwith * 3, 0);
			bb.y = MAX(ringCircles[i].outter.y - ringCircles[i].outter.bbheight * 3, 0);
			bb.width = (ringCircles[i].outter.bbwith * 6);
			if ((bb.x + bb.width) >= srcwidth)
				bb.width = srcwidth - bb.x;
			bb.height = (ringCircles[i].outter.bbheight * 6);
			if ((bb.y + bb.height) >= srcheight)
				bb.height = srcheight - bb.y;

			cv::Mat roi;
			frame_gray(bb).copyTo(roi);

	/*		cv::imshow("roi", roi);
			cv::waitKey(0);*/

			cv::Mat binary;
			cv::adaptiveThreshold(roi, binary, 255, cv::ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 7, 0);
#if 1
			// image erode and dialate
			{
				int erosion_type = cv::MORPH_RECT;
				int winSize = 3;
				cv::Mat element = cv::getStructuringElement(erosion_type, cv::Size(winSize, winSize),
					cv::Point(-1, -1));
				/// Apply the erosion operation
				erode(binary, binary, element);
				int dilation_type = cv::MORPH_RECT;
				dilate(binary, binary, element);
			}
#endif
			std::vector<std::vector<cv::Point>> contoursoi;
			std::vector<cv::Vec4i> hierarchy;
			cv::findContours(binary, contoursoi, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE,cv::Point(bb.x,bb.y));

			contours_openmp[id].insert(contours_openmp[id].end(), contoursoi.begin(), contoursoi.end());
		}
		t3 = std::chrono::high_resolution_clock::now();

		for (auto cts : contours_openmp)
			contours.insert(contours.end(), cts.begin(), cts.end());

		std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t1).count() << std::endl;
	}

	int minContourSize = 100 * resizeScale;
	int maxContourSize = 1200 * resizeScale;
	const double roundnessTolerance = 0.8;
	const double circularityTolerance = 0.3;
	const float centerDistanceToleranceAbs = 10;
	const float areaRatio = innerdiameter*innerdiameter / outterdiameter*outterdiameter;
	const float areaRatioTolerance = 0.3;
	const int boardMargin = 20;

	std::vector<int> can1;
	std::vector<__ellipseFeatures_t> canMoments;

#define USE_PARALLEL	0

#if USE_PARALLEL
	cv::setNumThreads(1);
	std:vector<uint8_t> valid(contours.size(), 0);
	ParallelProcessing pprocessing(contours, valid, resizeScale_inv, minContourSize, maxContourSize,roundnessTolerance);
	int length = contours.size();
	cv::parallel_for_(cv::Range(0, length), pprocessing);

	for (size_t i = 0; i < valid.size(); i++)
	{
		if (valid[i] == 1)
		{
			can1.push_back(i);
		}
	}

#else
	// do contours selection, roundness and contour's size, TODO, if we want this to work in large scale env, 
	//  we need to capture large dataset in order to determine the min, max contour size
#if PARALLEL_COMPUTING
#pragma omp parallel for
#endif
	for (int i = 0; i < contours.size(); i++)
	{
		const std::vector<cv::Point> &contour = contours[i];
		if (contour.size() > minContourSize && contour.size() < maxContourSize)
		{
			double perimeter = cv::arcLength(contour, true);
			double area = cv::contourArea(contour, false);
			double roundness = 4 * CV_PI*area / perimeter / perimeter;
			double errRoundness = fabs(roundness - 1);
			cv::Rect bb = cv::boundingRect(contour);
			
			if (bb.x < boardMargin || bb.y < boardMargin 
				|| (bb.x + bb.width) >= srcwidth * resizeScale 
				|| (bb.y + bb.height) >= resizeScale * srcheight)
				continue;

			if (errRoundness < roundnessTolerance)
			{
				//compute ellipse parameters
				__ellipseFeatures_t e;
#if ESTIMATE_ELLIPSE_BY_MOMENTS
				cv::Moments mm = cv::moments(contour);

				/* opencv, normalized moments exist some bugs */
				float scaleNomr = 1 / mm.m00;
				float nu11 = mm.mu11 * scaleNomr;
				float nu20 = mm.mu20 * scaleNomr;
				float nu02 = mm.mu02 * scaleNomr;

				float trace = nu20 + nu02;
				float det = 4 * nu11*nu11 + (nu20 - nu02) * (nu20 - nu02);
				if (det > 0) det = sqrt(det); else det = 0;//yes, that can happen as well:(
				float f0 = (trace + det) / 2;
				float f1 = (trace - det) / 2;

				/* cnetroid */
				e.x = mm.m10 / mm.m00 * resizeScale_inv;
				e.y = mm.m01 / mm.m00 * resizeScale_inv;
				/* major and minor axis */
				e.a = 2 * sqrt(f0) * resizeScale_inv;
				e.b = 2 * sqrt(f1) * resizeScale_inv;

				float dem = nu11*nu11 + (nu20 - f0)*(nu20 - f0);
				float sdem = sqrt(dem);

				if (nu11 != 0) {                               //aligned ?
					e.v0 = -nu11 / sdem; // no-> standard calculatiion
					e.v1 = (nu20 - f0) / sdem;
				}
				else {
					e.v0 = e.v1 = 0;
					if (nu20 > nu02) e.v0 = 1.0; else e.v1 = 1.0;   // aligned, hm, is is aligned with x or y ?
				}
				e.theta = 0.5 * atan2(2 * nu11, nu20 - nu02);
				e.area = area * resizeScale_inv * resizeScale_inv;

				e.bbwith = bb.width;
				e.bbheight = bb.height;
#else /* direct ellipse fitting */
				e.bbwith = bb.width;
				e.bbheight = bb.height;
				auto transform = [](const std::vector<cv::Point> &src) {
					std::vector<cv::Point2f> dst;
					for (auto pt : src)
						dst.push_back(cv::Point2f(pt.x, pt.y));
					return dst;
				};
				std::vector<cv::Point2f> srcPts;
				srcPts = transform(contour);
				std::vector<cv::Point2f> undistPts;
				//undistPts = srcPts;
				cv::undistortPoints(srcPts, undistPts, camK, distCoeff);
				for (int jj = 0; jj < undistPts.size(); jj++)
				{
					float x, y, z;
					x = camK.at<double>(0, 0)*undistPts[jj].x;// +camK.at<double>(0, 2);
					y = camK.at<double>(1, 1)*undistPts[jj].y;// +camK.at<double>(1, 2);
					undistPts[jj].x = x;
					undistPts[jj].y = y;
				}
				cv::Mat et(1, 6, CV_64F);
				cv::RotatedRect rrect = fitEllipseDirect(undistPts, et);

				e.a = rrect.size.width * 0.5 * resizeScale_inv;
				e.b = rrect.size.height * 0.5 * resizeScale_inv;
				e.x = rrect.center.x * resizeScale_inv + camK.at<double>(0, 2);
				e.y = rrect.center.y * resizeScale_inv + camK.at<double>(1, 2);

#if 1
				// subpixel interpolation
				float initx = e.x;
				float inity = e.y;
				int num_of_pixels = 4 * sqrt(e.a*e.a + e.b*e.b);
				if (num_of_pixels > contour.size())
				{
					num_of_pixels = contour.size();
				}
				// 
				for (int jj = 0; jj < contour.size(); jj++)
				{
					cv::Vec2f grad = cv::Vec2f(contour[jj].x - e.x, contour[jj].y - e.y);
					grad = grad / cv::norm(grad);
					//std::cout << grad << std::endl;
					cv::Vec2f pf1 = cv::Vec2f(contour[jj].x - 2 * grad[0], contour[jj].y - 2 * grad[1]);
					cv::Vec2f pf2 = cv::Vec2f(contour[jj].x - grad[0], contour[jj].y - grad[1]);
					cv::Vec2f pf4 = cv::Vec2f(contour[jj].x + 1*grad[0], contour[jj].y + 1*grad[1]);
					cv::Vec2f pf5 = cv::Vec2f(contour[jj].x + 2 * grad[0], contour[jj].y + 2 * grad[1]);

					// do bilinear interpolation
#define BILINEAR_INTER(ptf, intensity, height, width, isboard) \
{ \
	if (int(ptf[1]) >= height || int(ptf[0]) >= width || int(ptf[1]+1) >= height || int(ptf[0]+1) >= width )\
		isboard = 1;\
	else{\
		const uchar *ptr = (uchar*)(&(frame_gray.ptr<uchar>(int(ptf[1]))[int(ptf[0])]));\
		const uchar *ptrn = (uchar*)(&(frame_gray.ptr<uchar>(int(ptf[1]) + 1)[int(ptf[0])])); \
		float d1 = ptf[0] - int(ptf[0]);\
		float d2 = ptf[1] - int(ptf[1]);\
		intensity = (1 - d1)*(1 - d2)*ptr[0] + d1*(1 - d2)*ptr[+1] + d2*(1 - d1)*ptrn[0] + d2*d1*ptrn[+1];\
	}\
}

#define NN_INTER(ptf, intensity) \
{\
	const uchar *ptr = (uchar*)(&(frame_gray.ptr<uchar>(int(ptf[1]))[int(ptf[0])])); \
	const uchar *ptrn = (uchar*)(&(frame_gray.ptr<uchar>(int(ptf[1]) + 1)[int(ptf[0])])); \
	float d1 = ptf[0] - int(ptf[0]); \
	float d2 = ptf[1] - int(ptf[1]); \
	if (d1 < 0.5 && d2 < 0.5)intensity = ptr[0];\
	else if (d2 > 0.5 && d1<0.5) intensity = ptr[+1];\
	else if (d2 < 0.5 && d1>0.5) intensity = ptrn[+0];\
	else intensity = ptrn[+1];\
}\

					float int1, int2, int3, int4, int5;
					int isBoard = 0;
					BILINEAR_INTER(pf1, int1, srcheight, srcwidth, isBoard); if (isBoard) continue;
					BILINEAR_INTER(pf2, int2, srcheight, srcwidth, isBoard); if (isBoard) continue;
					int3 = frame_gray.ptr<uchar>(contour[jj].y)[contour[jj].x];
					BILINEAR_INTER(pf4, int4, srcheight, srcwidth, isBoard); if (isBoard) continue;
					BILINEAR_INTER(pf5, int5, srcheight, srcwidth, isBoard); if (isBoard) continue;

					//cv::Vec2f pf6 = cv::Vec2f(contour[jj].x - 3 * grad[0], contour[jj].y - 3 * grad[1]);
					//cv::Vec2f pf7 = cv::Vec2f(contour[jj].x + 3 * grad[0], contour[jj].y + 3 * grad[1]);
					//float int6, int7;
					//BILINEAR_INTER(pf6, int6);
					//BILINEAR_INTER(pf7, int7);

					//float a, b, c, d;

					//a =  0.0278*int7 - 0.0278*int5 - 0.0278*int4 + 0.0278*int2 + 0.0278*int1 - 0.0278*int6;
					//b =  0.0663*int7 - 0.0102*int5 - 0.0561*int4 - 0.0561*int2 - 0.0102*int1 + 0.0663*int6;
					//c = -0.0873*int7 + 0.2659*int5 + 0.2302*int4 - 0.2302*int2 - 0.2659*int1 + 0.0873*int6;
					//d = -0.1429*int7 + 0.2143*int5 + 0.4286*int4 + 0.4286*int2 + 0.2143*int1 - 0.1429*int6;

					//int1 -= int3; int2 -= int3; int4 -= int3; int5 -= int3;

					/*
					    0.0833   -0.1667    0.1667   -0.0833
						0.1667   -0.1667   -0.1667    0.1667
					   -0.0833    0.6667   -0.6667    0.0833
					   -0.1667    0.6667    0.6667   -0.1667
					*/

					float grad1 = (int2 - int1) ;
					float grad2 = (int3 - int2) ;
					float grad3 = (int4 - int3) ;
					float denom = MAX(grad1 + grad3 - 2 * grad2, 1);
					float dsub = -(grad1 - grad3) / (denom * 2);

					//a =  0.0833*int5 - 0.1667*int4 + 0.1667*int2 - 0.0833*int1;
					//b =  0.1667*int5 - 0.1667*int4 - 0.1667*int2 + 0.1667*int1;
					//c = -0.0833*int5 + 0.6667*int4 - 0.6667*int2 + 0.0833*int1;
					//d = -0.1667*int5 + 0.6667*int4 + 0.6667*int2 - 0.1667*int1;

					////std::cout << int5 << " ; " << int4 << " ; "<< int3 << " ; " << int2 << " ; " << int1 << std::endl;
					//////std::cout << grad1 << " ; " << grad2 << " ; " << grad3 << std::endl;
					//
					//int rules = b*b - 3 * a*c;
					//if (rules > 0)continue;
					
					//dsub = -b/(3*a+1e-6);
					//if (fabs(dsub) > 5)continue;
					//std::cout << dsub << std::endl;
					srcPts[jj].x = srcPts[jj].x + dsub * grad[0];
					srcPts[jj].y = srcPts[jj].y + dsub * grad[1];
				}
				cv::undistortPoints(srcPts, undistPts, camK, distCoeff);
				for (int jj = 0; jj < undistPts.size(); jj++)
				{
					float x, y, z;
					x = camK.at<double>(0, 0)*undistPts[jj].x;// +camK.at<double>(0, 2);
					y = camK.at<double>(1, 1)*undistPts[jj].y;// +camK.at<double>(1, 2);
					undistPts[jj].x = x;
					undistPts[jj].y = y;
				}
				// refitting
				rrect = fitEllipseDirect(undistPts, et);
#endif

				e.a = rrect.size.width * 0.5 * resizeScale_inv;
				e.b = rrect.size.height * 0.5 * resizeScale_inv;
				e.x = rrect.center.x * resizeScale_inv + camK.at<double>(0, 2);
				e.y = rrect.center.y * resizeScale_inv + camK.at<double>(1, 2);

				e.theta = rrect.angle;
				e.v0 = cos(e.theta);
				e.v1 = sin(e.theta);
				e.area = area * resizeScale_inv * resizeScale_inv;
				e.A = et.at<double>(0, 0);
				e.B = et.at<double>(0, 1);
				e.C = et.at<double>(0, 2);
				e.D = et.at<double>(0, 3);
				e.E = et.at<double>(0, 4);
				e.F = et.at<double>(0, 5);
#endif

				float circularity = CV_PI * (e.a)*(e.b) / e.area;
				if (fabsf(circularity - 1) < circularityTolerance) {
#if PARALLEL_COMPUTING
#pragma omp critical (can1)
#endif
					can1.push_back(i);
#if PARALLEL_COMPUTING
#pragma omp critical (canMoments)
#endif
					canMoments.push_back(e);
					//return true;
					//cv::drawContours(binaryc, contours, i, cv::Scalar(0, 255, 0, 0), 2, 8);
					//cv::namedWindow("binary", CV_WINDOW_NORMAL);
					//cv::imshow("binary", binaryc);
					//cv::waitKey(0);
				}
			}
		}
	}
#endif

	// inner/outter check
	std::vector<bool> matched(can1.size(), false);
	std::vector<ringCircularPattern> ringCircleCandiates;
#if PARALLEL_COMPUTING
#pragma omp parallel for
#endif
	for (int i = 0; i < can1.size(); i++)
	{
		if (matched[i])continue;
		__ellipseFeatures_t mmi = canMoments[i];
		for (size_t j = i + 1; j < can1.size(); j++)
		{
			__ellipseFeatures_t mmj = canMoments[j];
			float d = fabs(mmi.x - mmj.x) + fabs(mmi.y - mmj.y);
			float i2j = mmi.area / mmj.area;
			float j2i = 1 / i2j;

			float ratio1 = fabs(i2j - areaRatio);
			float ratio2 = fabs(j2i - areaRatio);

			if ((ratio1 < areaRatioTolerance || ratio2 < areaRatioTolerance)
				&& (d < centerDistanceToleranceAbs))
			{
#if 0
				/* pixel leakage correction */
				float r = areaRatio;
				float m0o, m1o, m0i, m1i;
				float ratio;
				if (mmi.area > mmj.area)
				{
					ratio = mmj.area / mmi.area;
					m0o = sqrt(mmi.a);
					m1o = sqrt(mmi.b);
					m0i = sqrt(ratio)*m0o;
					m1i = sqrt(ratio)*m1o;
				}
				else
				{
					ratio = mmi.area / mmj.area;
					m0o = sqrt(mmj.a);
					m1o = sqrt(mmj.b);
					m0i = sqrt(ratio)*m0o;
					m1i = sqrt(ratio)*m1o;
				}
				float a = (1 - r);
				float b = -(m0i + m1i) - (m0o + m1o)*r;
				float c = (m0i*m1i) - (m0o*m1o)*r;
				float t = (-b - sqrt(b*b - 4 * a*c)) / (2 * a);

				if (mmi.area > mmj.area)
				{
					mmj.a -= t;
					mmj.b -= t;
					mmi.a += t;
					mmi.b += t;
					ringCircleCandiates.push_back(ringCircularPattern(std::make_pair(can1[i], can1[j]), mmj, mmi));
				}
				else
				{
					mmi.a -= t;
					mmi.b -= t;
					mmj.a += t;
					mmj.b += t;
					ringCircleCandiates.push_back(ringCircularPattern(std::make_pair(can1[j], can1[i]), mmi, mmj));
				}
				matched[i] = true;
				matched[j] = true;
#else

#if 0		// do circle center refinement
				if (mmi.area > mmj.area)
				{
					float inv_f = 1. / camK.at<double>(0, 0);
					cv::Mat Qout(3, 3, CV_32F);
					cv::Mat Qin(3, 3, CV_32F);
					Qout.ptr<float>(0)[0] = mmi.A; Qout.ptr<float>(0)[1] = mmi.B*0.5; Qout.ptr<float>(0)[2] = mmi.D*0.5;
					Qout.ptr<float>(1)[0] = mmi.B*0.5; Qout.ptr<float>(1)[1] = mmi.C; Qout.ptr<float>(1)[2] = mmi.E*0.5;
					Qout.ptr<float>(2)[0] = mmi.D*0.5*inv_f; Qout.ptr<float>(2)[1] = mmi.E*0.5*inv_f; Qout.ptr<float>(2)[2] = mmi.F;

					Qin.ptr<float>(0)[0] = mmj.A; Qin.ptr<float>(0)[1] = mmj.B*0.5; Qin.ptr<float>(0)[2] = mmj.D*0.5;
					Qin.ptr<float>(1)[0] = mmj.B*0.5; Qin.ptr<float>(1)[1] = mmj.C; Qin.ptr<float>(1)[2] = mmj.E*0.5;
					Qin.ptr<float>(2)[0] = mmj.D*0.5; Qin.ptr<float>(2)[1] = mmj.E*0.5; Qin.ptr<float>(2)[2] = mmj.F;

					cv::Mat QouQininv = Qin * Qout.inv();
					cv::Mat eval, evecs;
					cv::eigen(QouQininv, eval, evecs);
					//std::cout << eval << std::endl;

					float lamda;
					int valid = 0;
					int rnum1 = 0, rnum2 = 1;

					if (fabs(eval.at<float>(0, 0) - eval.at<float>(1, 0)) < 0.25)
					{
						lamda = (eval.at<float>(0, 0) + eval.at<float>(1, 0))*0.5;
						valid = 1;
					}
					else if (fabs(eval.at<float>(1, 0) - eval.at<float>(2, 0)) < 0.25)
					{
						lamda = (eval.at<float>(1, 0) + eval.at<float>(2, 0))*0.5;
						rnum1 = 1;
						rnum2 = 2;
						valid = 1;
					}
					else if (fabs(eval.at<float>(2, 0) - eval.at<float>(0, 0)) < 0.25)
					{
						lamda = (eval.at<float>(0, 0) + eval.at<float>(2, 0))*0.5;
						//lamda = eval.at<float>(2, 0);
						rnum1 = 2;
						rnum2 = 0;
						valid = 1;
					}

					if (valid == 1)
					{
						cv::Mat Qsub = Qout.inv() - lamda * Qin.inv();

						std::cout << Qsub << std::endl;

						cv::Mat rc = Qsub.row(rnum1);
						float rnorm = rc.at<float>(0, 0)* rc.at<float>(0, 0) + rc.at<float>(0, 1)* rc.at<float>(0, 1) + rc.at<float>(0, 2)*rc.at<float>(0, 2);
						if (rnorm > 0.1)
						{
							mmi.x = mmj.x = rc.at<float>(0, 0) / rc.at<float>(0, 2) + camK.at<double>(0, 2);
							mmi.y = mmj.y = rc.at<float>(0, 1) / rc.at<float>(0, 2) + camK.at<double>(1, 2);
							mmi.a = mmi.a / fabs(lamda);
						}
						else
						{
							rc = Qsub.row(rnum2);
							rnorm = rc.at<float>(0, 0)* rc.at<float>(0, 0) + rc.at<float>(0, 1)* rc.at<float>(0, 1) + rc.at<float>(0, 2)*rc.at<float>(0, 2);
							if (rnorm > 0.1)
							{
								mmi.x = mmj.x = rc.at<float>(0, 0) / rc.at<float>(0, 2) * camK.at<double>(0, 0) + camK.at<double>(0, 2);
								mmi.y = mmj.y = rc.at<float>(0, 1) / rc.at<float>(0, 2) * camK.at<double>(0, 0) + camK.at<double>(1, 2);
								mmi.a = mmi.a / fabs(lamda);
							}
						}
					}
				}
				else
				{
					float inv_f = 1. / camK.at<double>(0, 0);
					cv::Mat Qout(3, 3, CV_32F);
					cv::Mat Qin(3, 3, CV_32F);
					Qout.ptr<float>(0)[0] = mmj.A; Qout.ptr<float>(0)[1] = mmj.B*0.5; Qout.ptr<float>(0)[2] = mmj.D*0.5;
					Qout.ptr<float>(1)[0] = mmj.B*0.5; Qout.ptr<float>(1)[1] = mmj.C; Qout.ptr<float>(1)[2] = mmj.E*0.5;
					Qout.ptr<float>(2)[0] = mmj.D*0.5; Qout.ptr<float>(2)[1] = mmj.E*0.5; Qout.ptr<float>(2)[2] = mmj.F;

					Qin.ptr<float>(0)[0] = mmi.A; Qin.ptr<float>(0)[1] = mmi.B*0.5; Qin.ptr<float>(0)[2] = mmi.D*0.5;
					Qin.ptr<float>(1)[0] = mmi.B*0.5; Qin.ptr<float>(1)[1] = mmi.C; Qin.ptr<float>(1)[2] = mmi.E*0.5;
					Qin.ptr<float>(2)[0] = mmi.D*0.5; Qin.ptr<float>(2)[1] = mmi.E*0.5; Qin.ptr<float>(2)[2] = mmi.F;

					cv::Mat QouQininv = Qout * Qin.inv();
					cv::Mat eval, evecs;
					cv::eigen(QouQininv, eval, evecs);
					std::cout << eval << std::endl;

					float lamda;
					int valid = 0;
					int rnum1 = 0, rnum2 = 1;

					if (fabs(eval.at<float>(0, 0) - eval.at<float>(1, 0)) < 0.25)
					{
						lamda = (eval.at<float>(0, 0) + eval.at<float>(1, 0))*0.5;
						valid = 1;
					}
					else if (fabs(eval.at<float>(1, 0) - eval.at<float>(2, 0)) < 0.25)
					{
						lamda = (eval.at<float>(1, 0) + eval.at<float>(2, 0))*0.5;
						rnum1 = 1;
						rnum2 = 2;
						valid = 1;
					}
					else if (fabs(eval.at<float>(2, 0) - eval.at<float>(0, 0)) < 0.25)
					{
						lamda = (eval.at<float>(0, 0) + eval.at<float>(2, 0))*0.5;
						rnum1 = 2;
						rnum2 = 0;
						valid = 1;
					}

					if (valid == 1)
					{
						cv::Mat Qsub = Qin.inv() - lamda * Qout.inv();

						std::cout << Qsub << std::endl;

						cv::Mat rc = Qsub.row(rnum1);
						float rnorm = rc.at<float>(0, 0)* rc.at<float>(0, 0) + rc.at<float>(0, 1)* rc.at<float>(0, 1) + rc.at<float>(0, 2)*rc.at<float>(0, 2);
						if (rnorm > 0.1)
						{
							std::cout << mmi.x << " ; " << mmj.x << " ; " << rc.at<float>(0, 0) / rc.at<float>(0, 2) + camK.at<double>(0, 2) << std::endl;
							std::cout << mmi.y << " ; " << mmj.y << " ; " << rc.at<float>(0, 1) / rc.at<float>(0, 2) + camK.at<double>(1, 2) << std::endl;

							mmi.x = mmj.x = rc.at<float>(0, 0) / rc.at<float>(0, 2) + camK.at<double>(0, 2);
							mmi.y = mmj.y = rc.at<float>(0, 1) / rc.at<float>(0, 2) + camK.at<double>(1, 2);
							mmj.a = mmj.a / fabs(lamda);

						}
						else
						{
							rc = Qsub.row(rnum2);
							rnorm = rc.at<float>(0, 0)* rc.at<float>(0, 0) + rc.at<float>(0, 1)* rc.at<float>(0, 1) + rc.at<float>(0, 2)*rc.at<float>(0, 2);
							if (rnorm > 0.1)
							{
								std::cout << mmi.x << " ; " << mmj.x << " ; " << rc.at<float>(0, 0) / rc.at<float>(0, 2) + camK.at<double>(0, 2) << std::endl;
								std::cout << mmi.y << " ; " << mmj.y << " ; " << rc.at<float>(0, 1) / rc.at<float>(0, 2) + camK.at<double>(1, 2) << std::endl;
								mmi.x = mmj.x = rc.at<float>(0, 0) / rc.at<float>(0, 2) + camK.at<double>(0, 2);
								mmi.y = mmj.y = rc.at<float>(0, 1) / rc.at<float>(0, 2) + camK.at<double>(1, 2);
								mmj.a = mmj.a / fabs(lamda);
							}
						}
					}
				}
#endif

				if (mmi.area > mmj.area)
				{
#if PARALLEL_COMPUTING
#pragma omp critical (ringCircleCandiates)
#endif
					ringCircleCandiates.push_back(ringCircularPattern(std::make_pair(can1[i], can1[j]), mmj, mmi));
				}
				else
				{
#if PARALLEL_COMPUTING
#pragma omp critical (ringCircleCandiates)
#endif
					ringCircleCandiates.push_back(ringCircularPattern(std::make_pair(can1[j], can1[i]), mmi, mmj));
				}
				matched[i] = true;
				matched[j] = true;
				break;
#endif
			}
		}
	}

	// check multiple responses
	ringCircles.clear();
#if PARALLEL_COMPUTING
#pragma omp parallel for
#endif
	for (int i = 0; i < ringCircleCandiates.size(); i++)
	{
		int xi = ringCircleCandiates[i].outter.x;
		int yi = ringCircleCandiates[i].outter.y;
		size_t j = 0;
		for (j = i+1; j < ringCircleCandiates.size(); j++)
		{
			int xj = ringCircleCandiates[j].outter.x;
			int yj = ringCircleCandiates[j].outter.y;
			if ((abs(xi-xj)+abs(yi-yj)) < 10)
			{
				break;
			}
		}
		if (j != ringCircleCandiates.size())//to close
			continue;
#if PARALLEL_COMPUTING
#pragma omp critical (ringCircles)
#endif
		ringCircles.push_back(ringCircleCandiates[i]);
	}
	
	std::sort(ringCircles.begin(), ringCircles.end(), comp);

	std::chrono::high_resolution_clock::time_point t4 = std::chrono::high_resolution_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count() << std::endl;
	//std::chrono::high_resolution_clock::time_point t5 = std::chrono::high_resolution_clock::now();
	//std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t4).count() << std::endl;

	
	//for (size_t i = 0; i < ringCircleCandiates.size(); i++)
	//{
	//	//cv::drawContours(binaryc, contours, i, cv::Scalar(0, 255, 0, 0), 2, 8);
	//	cv::drawContours(binaryc, contours, ringCircles[i].matchpair.first, cv::Scalar(0, 0, 255, 0), 1, 8);
	//	cv::drawContours(binaryc, contours, ringCircles[i].matchpair.second, cv::Scalar(255, 0, 0, 0), 1, 8);
	//}
	//cv::namedWindow("binary", CV_WINDOW_NORMAL);
	//cv::imshow("binary", binaryc);
	//cv::waitKey(0);

	return ringCircles.size();
}

void circularPatternBasedLocSystems::undistort(float x_in, float y_in, float& x_out, float& y_out)
{
#if defined(ENABLE_FULL_UNDISTORT)
	x = (x - cc[0]) / fc[0];
	y = (y - cc[1]) / fc[1];
#else
	std::vector<cv::Vec2f> src(1, cv::Vec2f(x_in, y_in));
	std::vector<cv::Vec2f> dst(1);
	cv::undistortPoints(src, dst, camK, distCoeff);
	x_out = dst[0](0); y_out = dst[0](1);
#endif
}

// Generic functor
template<typename _Scalar, int NX = Eigen::Dynamic, int NY = Eigen::Dynamic>
struct Functor
{
	typedef _Scalar Scalar;
	enum {
		InputsAtCompileTime = NX,
		ValuesAtCompileTime = NY
	};
	typedef Eigen::Matrix<Scalar, InputsAtCompileTime, 1> InputType;
	typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, 1> ValueType;
	typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, InputsAtCompileTime> JacobianType;

	int m_inputs, m_values;

	Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
	Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}

	int inputs() const { return m_inputs; }
	int values() const { return m_values; }

};

void getpose(float cx, float cy, float a, float b, float v0, float v1, float uu0, float vv0, float invf, float diameter,
	cv::Vec3f &position);

struct my_functor : Functor<double>
{
	my_functor(float v0, float v1, float uu0, float vv0, float invf, float ao, float bo, float ai, float bi, float diametero, float diameteri) : Functor<double>(3, 4),
		_v0(v0), _v1(v1), _uu0(uu0), _vv0(vv0), _invf(invf), _ao(ao), _bo(bo), _ai(ai), _bi(bi), _diametero(diametero), _diameteri(diameteri){}
	int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const
	{
		double cx = x(0);
		double cy = x(1);
		double _d = x(2);
		//std::cout << " optx " << x << std::endl;

		// for outter
		double ao = _ao + _d; double bo = _bo + _d;
		cv::Vec3f position1;
		getpose(cx, cy, ao, bo, _v0, _v1, _uu0, _vv0, _invf, _diametero, position1);
		//std::cout << " position1 " << position1 << std::endl;

		//for inner
		double ai = _ai - _d; double bi = _bi - _d;
		cv::Vec3f position2;
		getpose(cx, cy, ai, bi, _v0, _v1, _uu0, _vv0, _invf, _diameteri, position2);
		//std::cout << "position2" << position2 << std::endl;

		// Implement
		fvec(0) = position1[0] - position2[0];
		fvec(1) = position1[1] - position2[1];
		fvec(2) = position1[2] - position2[2];
		fvec(3) = (ai*bi) / (ao*bo) - (_diameteri*_diameteri) / (_diametero*_diametero);

		//std::cout << "fvec:" << fvec << std::endl;

		return 0;
	}

	float _v0;
	float _v1;
	float _uu0;
	float _vv0;
	float _invf;
	float _ao, _bo, _ai, _bi;
	float _diametero, _diameteri;
};


void getpose(float cx, float cy, float a, float b, float nv0, float nv1, float uu0, float vv0, float invf, float diameter,
	cv::Vec3f &position)
{
	float x, y, x1, x2, y1, y2, sx1, sx2, sy1, sy2, major, minor, v0, v1;
	//transform the center
	x = (cx - uu0)*invf;
	y = (cy - vv0)*invf;
	//calculate the major axis 
	//endpoints in image coords
	sx1 = cx + nv0 * a;
	sx2 = cx - nv0 * a;
	sy1 = cy + nv1 * a;
	sy2 = cy - nv1 * a;

	//endpoints in camera coords 
	x1 = (sx1 - uu0)*invf;; y1 = (sy1 - vv0)*invf;
	x2 = (sx2 - uu0)*invf;; y2 = (sy2 - vv0)*invf;

	//semiaxis length 
	major = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2)) / 2.0;

	v0 = (x2 - x1) / major / 2.0;
	v1 = (y2 - y1) / major / 2.0;

	//calculate the minor axis 
	//endpoints in image coords
	sx1 = cx + nv1 * b;
	sx2 = cx - nv1 * b;
	sy1 = cy - nv0 * b;
	sy2 = cy + nv0 * b;

	//endpoints in camera coords 
	x1 = (sx1 - uu0)*invf;; y1 = (sy1 - vv0)*invf;
	x2 = (sx2 - uu0)*invf;; y2 = (sy2 - vv0)*invf;

	//semiaxis length 
	minor = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2)) / 2.0;
	//construct the conic
	float A, B, C, D, E, F;
	A = v0*v0 / (major*major) + v1*v1 / (minor*minor);
	B = v0*v1*(1 / (major*major) - 1 / (minor*minor));
	C = v0*v0 / (minor*minor) + v1*v1 / (major*major);
	D = (-x*A - B*y);
	E = (-y*C - B*x);
	F = (A*x*x + C*y*y + 2 * B*x*y - 1);

	cv::Matx33d data(A, B, D,
		B, C, E,
		D, E, F);
	// compute conic eigenvalues and eigenvectors
	cv::Vec3d eigenvalues;
	cv::Matx33d eigenvectors;
	cv::eigen(data, eigenvalues, eigenvectors);
	// compute ellipse parameters in real-world
	double L1 = eigenvalues(1);
	double L2 = eigenvalues(0);
	double L3 = eigenvalues(2);
	int V2 = 0;
	int V3 = 2;

	// position
	double z = diameter / sqrt(-L2*L3) / 2.0;
	cv::Matx13d position_mat = L3 * sqrt((L2 - L1) / (L2 - L3)) * eigenvectors.row(V2)
		+ L2 * sqrt((L1 - L3) / (L2 - L3)) * eigenvectors.row(V3);
	position = cv::Vec3f(position_mat(0), position_mat(1), position_mat(2));
	int S3 = (position(2) * z < 0 ? -1 : 1);
	position *= S3 * z;
}

void circularPatternBasedLocSystems::getpos(ringCircularPattern &circle, 
	cv::Vec3f &position, cv::Vec3f rotation)
{
	float uu0 = camK.at<double>(0, 2);	float vv0 = camK.at<double>(1, 2);
	float invf = 1 / camK.at<double>(0, 0);

	// do a L-M optimization
	Eigen::VectorXd optx(3);
	optx(0) = circle.outter.x;
	optx(1) = circle.outter.y;
	optx(2) = 0;

	//std::cout << "x: " << optx << std::endl;

	my_functor functor(circle.outter.v0, circle.outter.v1, uu0, vv0, invf, circle.outter.a, circle.outter.b, circle.inner.a, circle.inner.b, outterdiameter, innerdiameter);
	Eigen::NumericalDiff<my_functor> numDiff(functor);
	Eigen::LevenbergMarquardt<Eigen::NumericalDiff<my_functor>, double> lm(numDiff);
	lm.parameters.maxfev = 100;
	lm.parameters.xtol = 1.0e-10;
	std::cout << lm.parameters.maxfev << std::endl;
	int ret = lm.minimize(optx);
	
	/*std::cout << lm.iter << std::endl;
	std::cout << ret << std::endl;
	std::cout << "x that minimizes the function: " << optx << std::endl;*/

	circle.outter.x = optx(0);
	circle.outter.y = optx(1);
	circle.outter.a = circle.outter.a + optx(2);
	circle.outter.b = circle.outter.b + optx(2);

	//cv::RotatedRect rect;
	//rect.angle = circle.outter.theta;
	//rect.center.x = circle.outter.x - camK.at<double>(0, 2);
	//rect.center.y = circle.outter.y - camK.at<double>(1, 2);
	//rect.size.width = circle.outter.a * 2;
	//rect.size.height = circle.outter.b * 2;

	//cv::Mat et(1, 6, CV_64F);
	//fromRotatedRectToEllipseParams(rect, et);

	//circle.outter.A = et.at<double>(0, 0);
	//circle.outter.B = et.at<double>(0, 1);
	//circle.outter.C = et.at<double>(0, 2);
	//circle.outter.D = et.at<double>(0, 3);
	//circle.outter.E = et.at<double>(0, 4);
	//circle.outter.F = et.at<double>(0, 5);

#if 1
	float x, y, x1, x2, y1, y2, sx1, sx2, sy1, sy2, major, minor, v0, v1;
	//transform the center
	//undistort(circle.outter.x, circle.outter.y, x, y);

	x = (circle.outter.x - uu0)*invf;
	y = (circle.outter.y - vv0)*invf;
	//calculate the major axis 
	//endpoints in image coords
	sx1 = circle.outter.x + circle.outter.v0 * circle.outter.a;
	sx2 = circle.outter.x - circle.outter.v0 * circle.outter.a;
	sy1 = circle.outter.y + circle.outter.v1 * circle.outter.a;
	sy2 = circle.outter.y - circle.outter.v1 * circle.outter.a;

	//endpoints in camera coords 
	//undistort(sx1, sy1, x1, y1);
	//undistort(sx2, sy2, x2, y2);
	x1 = (sx1 - uu0)*invf;; y1 = (sy1 - vv0)*invf;
	x2 = (sx2 - uu0)*invf;; y2 = (sy2 - vv0)*invf;

	//semiaxis length 
	major = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2)) / 2.0;

	v0 = (x2 - x1) / major / 2.0;
	v1 = (y2 - y1) / major / 2.0;

	//calculate the minor axis 
	//endpoints in image coords
	sx1 = circle.outter.x + circle.outter.v1 * circle.outter.b;
	sx2 = circle.outter.x - circle.outter.v1 * circle.outter.b;
	sy1 = circle.outter.y - circle.outter.v0 * circle.outter.b;
	sy2 = circle.outter.y + circle.outter.v0 * circle.outter.b;

	//endpoints in camera coords 
	//undistort(sx1, sy1, x1, y1);
	//undistort(sx2, sy2, x2, y2);
	x1 = (sx1 - uu0)*invf;; y1 = (sy1 - vv0)*invf;
	x2 = (sx2 - uu0)*invf;; y2 = (sy2 - vv0)*invf;

	//semiaxis length 
	minor = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2)) / 2.0;
	//construct the conic
	float a, b, c, d, e, f;
	a = v0*v0 / (major*major) + v1*v1 / (minor*minor);
	b = v0*v1*(1 / (major*major) - 1 / (minor*minor));
	c = v0*v0 / (minor*minor) + v1*v1 / (major*major);
	d = (-x*a - b*y);
	e = (-y*c - b*x);
	f = (a*x*x + c*y*y + 2 * b*x*y - 1);

	cv::Matx33d data(a, b, d,
		b, c, e,
		d, e, f);
	// compute conic eigenvalues and eigenvectors
	cv::Vec3d eigenvalues;
	cv::Matx33d eigenvectors;
	cv::eigen(data, eigenvalues, eigenvectors);
	// compute ellipse parameters in real-world
	double L1 = eigenvalues(1);
	double L2 = eigenvalues(0);
	double L3 = eigenvalues(2);
	int V2 = 0;
	int V3 = 2;

	// position
	double z = outterdiameter / sqrt(-L2*L3) / 2.0;
	cv::Matx13d position_mat = L3 * sqrt((L2 - L1) / (L2 - L3)) * eigenvectors.row(V2)
		+ L2 * sqrt((L1 - L3) / (L2 - L3)) * eigenvectors.row(V3);
	position = cv::Vec3f(position_mat(0), position_mat(1), position_mat(2));
	int S3 = (position(2) * z < 0 ? -1 : 1);
	position *= S3 * z;

	rotation(0) = acos(circle.outter.b / circle.outter.a) / CV_PI*180.0;
	rotation(1) = atan2(circle.outter.v1, circle.outter.v0) / CV_PI*180.0;
	rotation(2) = circle.outter.v1 / circle.outter.v0;


	//cv::Matx13d n1 = sqrt((L2 - L1) / (L2 - L3))*eigenvectors.row(V2) - sqrt((L1 - L3) / (L2 - L3))*eigenvectors.row(V3);
	cv::Matx13d n2 = sqrt((L2 - L1) / (L2 - L3))*eigenvectors.row(V2) + sqrt((L1 - L3) / (L2 - L3))*eigenvectors.row(V3);
	//cv::Matx13d n3 = -sqrt((L2 - L1) / (L2 - L3))*eigenvectors.row(V2) - sqrt((L1 - L3) / (L2 - L3))*eigenvectors.row(V3);
	//cv::Matx13d n4 = -sqrt((L2 - L1) / (L2 - L3))*eigenvectors.row(V2) + sqrt((L1 - L3) / (L2 - L3))*eigenvectors.row(V3);
	if (n2(2) < 0)
	{
		n2 = n2 * -1;
	}

	//std::cout << n2 << std::endl;
	circle.r33.at<float>(0, 2) = n2(0);
	circle.r33.at<float>(1, 2) = n2(1);
	circle.r33.at<float>(2, 2) = n2(2);

#else
	//cv::RotatedRect rect;
	//rect.angle = circle.outter.theta;
	//rect.center.x = circle.outter.x;
	//rect.center.y = circle.outter.y;
	//rect.size.width = circle.outter.a * 2;
	//rect.size.height = circle.outter.b * 2;
	
	//cv::Mat et(1, 6, CV_64F);
	//fromRotatedRectToEllipseParams(rect, et);

	//et.at<double>(0, 0) = circle.outter.A;
	//et.at<double>(0, 1) = circle.outter.B;
	//et.at<double>(0, 2) = circle.outter.C;
	//et.at<double>(0, 3) = circle.outter.D;
	//et.at<double>(0, 4) = circle.outter.E;
	//et.at<double>(0, 5) = circle.outter.F;

	position = estimatePositionAnalyticalSol(et, camK, outterdiameter);

	rotation(0) = 0;
	rotation(1) = 0;
	rotation(2) = 0;
#endif
}

void circularPatternBasedLocSystems::localization()
{
#if PARALLEL_COMPUTING
//#pragma omp parallel for // not sure about eigen nonlinear optimizer works compatitble with omp parallel 
#endif
	for (int i = 0; i < ringCircles.size(); i++)
	{
		cv::Vec3f pos;
		cv::Vec3f rot;
		getpos(ringCircles[i], pos, rot);

		// do transform, assume planar, use homography
		cv::Vec3f post = pos;
		
		if (setAxis == true)
		{
			post = coordinateTransform * pos;
			post(0) /= post(2);
			post(1) /= post(2);
			post(2) = 0;
		}
		// TODO, use multiple circles to determine the accurate rotation
		ringCircles[i].t = post;
	}
}

struct reproj_functor : Functor<double>
{
	reproj_functor(cv::Mat &r33, float diameteroutter, float f,
		double A1, double B1, double C1, double D1, double E1, double F1,
		double A2, double B2, double C2, double D2, double E2, double F2, double interdist) : Functor<double>(6, 6),
		_r33(r33), _radius(diameteroutter*0.5), _f(f),
		_QA1(A1), _QB1(B1), _QC1(C1), _QD1(D1), _QE1(E1), _QF1(F1),
		_QA2(A2), _QB2(B2), _QC2(C2), _QD2(D2), _QE2(E2), _QF2(F2), _interdist(interdist){}
	int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const
	{
		double x1 = x(0);
		double y1 = x(1);
		double z1 = x(2);
		double x2 = x(3);
		double y2 = x(4);
		double z2 = x(5);

		cv::Vec3f ds = cv::Vec3f(x2 - x1, y2 - y1, z2 - z1);

#if 0
		cv::Vec3f ai = ds / cv::norm(ds);
		cv::Vec3f ak;
		ak(0) = _r33.at<float>(0, 2);
		ak(1) = _r33.at<float>(1, 2);
		ak(2) = _r33.at<float>(2, 2);
		cv::Vec3f aj = ak.cross(ai);
		cv::Mat r(3, 3, CV_32F);
		r.ptr<float>(0)[0] = ai(0); r.ptr<float>(0)[1] = aj(0); r.ptr<float>(0)[2] = ak(0);
		r.ptr<float>(1)[0] = ai(1); r.ptr<float>(1)[1] = aj(1); r.ptr<float>(1)[2] = ak(1);
		r.ptr<float>(2)[0] = ai(2); r.ptr<float>(2)[1] = aj(2); r.ptr<float>(2)[2] = ak(2);
		//r = r.t();
		// constant points
		static const double cpp[8][2] = {
			{ _radius,0},{ _radius * sqrt(2) / 2, _radius * sqrt(2) / 2},{ 0, _radius },{ -_radius * sqrt(2) / 2, _radius * sqrt(2) / 2 },
			{ -_radius,0 },{ -_radius * sqrt(2) / 2, -_radius * sqrt(2) / 2 },{ 0, -_radius },{ _radius * sqrt(2) / 2, -_radius * sqrt(2) / 2 } };
				
		float *ptrr = r.ptr<float>(0);
		// for 2 circles
		for (int i = 0; i < 8; i++)
		{
			double xc1 = ptrr[0] * cpp[i][0] + ptrr[1] * cpp[i][1] - x1;
			double yc1 = ptrr[3] * cpp[i][0] + ptrr[4] * cpp[i][1] - y1;
			double zc1 = ptrr[6] * cpp[i][0] + ptrr[7] * cpp[i][1] - z1;

			double xc2 = ptrr[0] * cpp[i][0] + ptrr[1] * cpp[i][1] - x2;
			double yc2 = ptrr[3] * cpp[i][0] + ptrr[4] * cpp[i][1] - y2;
			double zc2 = ptrr[6] * cpp[i][0] + ptrr[7] * cpp[i][1] - z2;

			//projection
			float u1 = xc1 / zc1 * _f; float v1 = yc1 / zc1 * _f;
			float u2 = xc2 / zc2 * _f; float v2 = yc2 / zc2 * _f;

			fvec(i) = _QA1*u1*u1 + _QB1*u1*v1 + _QC1*v1*v1 + _QD1*u1 + _QE1*v1 + _QF1;
			fvec(i+8) = _QA2*u2*u2 + _QB2*u2*v2 + _QC2*v2*v2 + _QD2*u2 + _QE2*v2 + _QF2;
		}
#endif
		fvec(0) = cv::norm(ds) - _interdist;
		fvec(1) = fvec(2) = fvec(3) = fvec(4) = fvec(5) = 0;
		//std::cout << "fvec:" << fvec << std::endl;

		return 0;
	}

	float _radius;
	cv::Mat3f _r33;
	float _f;
	double _QA1, _QB1, _QC1, _QD1, _QE1, _QF1, _QA2, _QB2, _QC2, _QD2, _QE2, _QF2;
	double _interdist;
};


void circularPatternBasedLocSystems::routineFullPoseEstimationBasedOn2Markers(cv::Mat &src, float interdistance)
{
	if (ringCircles.size() != 2)
		return;

	cv::Vec3f ds = ringCircles[1].t - ringCircles[0].t;
	float ds_norm = cv::norm(ds);
	
	if (fabs(ds_norm - interdistance) > 0.3)// too error
		return;

	// correct
	// first, estimate r;
	cv::Vec3f ai = ds / ds_norm;
	cv::Vec3f ak;
	ak(0) = ringCircles[0].r33.at<float>(0, 2);	
	ak(1) = ringCircles[0].r33.at<float>(1, 2);
	ak(2) = ringCircles[0].r33.at<float>(2, 2);
	cv::Vec3f aj = ak.cross(ai);
	cv::Mat r(3,3,CV_32F);
	r.ptr<float>(0)[0] = ai(0); r.ptr<float>(0)[1] = aj(0); r.ptr<float>(0)[2] = ak(0);
	r.ptr<float>(1)[0] = ai(1); r.ptr<float>(1)[1] = aj(1); r.ptr<float>(1)[2] = ak(1);
	r.ptr<float>(2)[0] = ai(2); r.ptr<float>(2)[1] = aj(2); r.ptr<float>(2)[2] = ak(2);

#if 0
	float _radius = outterdiameter * 0.5;
	static const double cpp[8][2] = {
		{ _radius,0 },{ _radius * sqrt(2) / 2, _radius * sqrt(2) / 2 },{ 0, _radius },{ -_radius * sqrt(2) / 2, _radius * sqrt(2) / 2 },
		{ -_radius,0 },{ -_radius * sqrt(2) / 2, -_radius * sqrt(2) / 2 },{ 0, -_radius },{ _radius * sqrt(2) / 2, -_radius * sqrt(2) / 2 } };
	//std::cout << r << std::endl;
	//r = r.t();
	//std::cout << r << std::endl;
	float *ptrr = r.ptr<float>(0);
	// for 2 circles
	float _f = camK.at<double>(0, 0);
	float _u0 = camK.at<double>(0, 2);
	float _v0 = camK.at<double>(1, 2);
	for (int i = 0; i < 8; i++)
	{
		double xc1 = ptrr[0] * cpp[i][0] + ptrr[1] * cpp[i][1] - ringCircles[0].t(0);
		double yc1 = ptrr[3] * cpp[i][0] + ptrr[4] * cpp[i][1] - ringCircles[0].t(1);
		double zc1 = ptrr[6] * cpp[i][0] + ptrr[7] * cpp[i][1] - ringCircles[0].t(2);

		double xc2 = ptrr[0] * cpp[i][0] + ptrr[1] * cpp[i][1] - ringCircles[1].t(0);
		double yc2 = ptrr[3] * cpp[i][0] + ptrr[4] * cpp[i][1] - ringCircles[1].t(1);
		double zc2 = ptrr[6] * cpp[i][0] + ptrr[7] * cpp[i][1] - ringCircles[1].t(2);

		//projection
		float u1 = xc1 / zc1 * _f; float v1 = yc1 / zc1 * _f;
		float u2 = xc2 / zc2 * _f; float v2 = yc2 / zc2 * _f;

		cv::circle(src, cv::Point(u1+ _u0, v1+ _v0), 2, cv::Scalar(0, 0, 255, 0), 2, 8, 0);
		cv::circle(src, cv::Point(u2+ _u0, v2+ _v0), 2, cv::Scalar(0, 255, 0, 0), 2, 8, 0);
	}
	cv::namedWindow("src", CV_WINDOW_NORMAL);
	cv::resizeWindow("src", 640, 480);
	cv::imshow("src", src);
	cv::waitKey(10);
	return;
#endif

	// do a L-M optimization
	Eigen::VectorXd optx(6);
	optx(0) = ringCircles[0].t(0);
	optx(1) = ringCircles[0].t(1);
	optx(2) = ringCircles[0].t(2);
	optx(3) = ringCircles[1].t(0);
	optx(4) = ringCircles[1].t(1);
	optx(5) = ringCircles[1].t(2);

	std::cout << "x: " << ringCircles[0].outter.x - camK.at<double>(0,2) << "y: " << ringCircles[0].outter.y - camK.at<double>(1, 2) << std::endl;
	std::cout << "x: " << ringCircles[1].outter.x - camK.at<double>(0, 2) << "y: " << ringCircles[1].outter.y - camK.at<double>(1, 2) << std::endl;

	std::cout << "x: " << optx << std::endl;

	reproj_functor functor(r,outterdiameter,camK.at<double>(0,0),
		ringCircles[0].outter.A,ringCircles[0].outter.B, ringCircles[0].outter.C, ringCircles[0].outter.D, ringCircles[0].outter.E, ringCircles[0].outter.F,
		ringCircles[1].outter.A, ringCircles[1].outter.B, ringCircles[1].outter.C, ringCircles[1].outter.D, ringCircles[1].outter.E, ringCircles[1].outter.F, interdistance);

	Eigen::NumericalDiff<reproj_functor> numDiff(functor);
	Eigen::LevenbergMarquardt<Eigen::NumericalDiff<reproj_functor>, double> lm(numDiff);
	lm.parameters.maxfev = 20;
	lm.parameters.xtol = 1.0e-10;
	std::cout << lm.parameters.maxfev << std::endl;
	int ret = lm.minimize(optx);

	std::cout << lm.iter << std::endl;
	std::cout << ret << std::endl;
	std::cout << "x that minimizes the function: " << optx << std::endl; 

	ringCircles[1].t = cv::Vec3f(optx(3), optx(4), optx(5));
	ringCircles[0].t = cv::Vec3f(optx(0), optx(1), optx(2));

	ds = ringCircles[1].t - ringCircles[0].t;
	ai = ds / ds_norm;
	aj = ak.cross(ai);
	r.ptr<float>(0)[0] = ai(0); r.ptr<float>(0)[1] = aj(0); r.ptr<float>(0)[2] = ak(0);
	r.ptr<float>(1)[0] = ai(1); r.ptr<float>(1)[1] = aj(1); r.ptr<float>(1)[2] = ak(1);
	r.ptr<float>(2)[0] = ai(2); r.ptr<float>(2)[1] = aj(2); r.ptr<float>(2)[2] = ak(2);

	ringCircles[0].r33 = r;
	ringCircles[1].r33 = r;

	std::cout << ringCircles[0].t << std::endl;
	std::cout << ringCircles[1].t << std::endl;
	std::cout << ringCircles[0].r33 << std::endl;
}



struct marker3_functor : Functor<double>
{
	marker3_functor(double hinterdist, double vinterdist) : Functor<double>(9, 9),
		_hinterdist(hinterdist), _vinterdist(vinterdist){}
	int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const
	{
		double x1 = x(0);
		double y1 = x(1);
		double z1 = x(2);
		double x2 = x(3);
		double y2 = x(4);
		double z2 = x(5);
		double x3 = x(6);
		double y3 = x(7);
		double z3 = x(8);

		cv::Vec3f dsh = cv::Vec3f(x2 - x1, y2 - y1, z2 - z1);
		cv::Vec3f dsv = cv::Vec3f(x3 - x1, y3 - y1, z3 - z1);

		fvec(0) = cv::norm(dsh) - _hinterdist;
		fvec(1) = cv::norm(dsv) - _vinterdist;
		fvec(2) = dsh.dot(dsv) / cv::norm(dsh) / cv::norm(dsv);
		fvec(3) = fvec(4) = fvec(5) = fvec(6) = fvec(7) = fvec(8) = 0;
		//std::cout << "fvec:" << fvec << std::endl;

		return 0;
	}

	double _hinterdist,_vinterdist;
};

void circularPatternBasedLocSystems::routineFullPoseEstimationBasedOn3Markers(cv::Mat &src, float hinterdistance, float vinterdistance)
{
	if (ringCircles.size() != 3)
		return;

	cv::Vec3f ds1 = ringCircles[1].t - ringCircles[0].t;
	cv::Vec3f ds2 = ringCircles[2].t - ringCircles[0].t;
	cv::Vec3f ds3 = ringCircles[2].t - ringCircles[1].t;

	float ds1_norm = cv::norm(ds1);
	float ds2_norm = cv::norm(ds2);
	float ds3_norm = cv::norm(ds3);

	int origo_id = -1, x_id, y_id;
	if (fabs(ds1_norm - hinterdistance) < 0.03)
	{
		if (fabs(ds2_norm - vinterdistance) < 0.03)
		{
			origo_id = 0; x_id = 1; y_id = 2;
		}
		else if (fabs(ds3_norm - vinterdistance) < 0.03)
		{
			origo_id = 1; x_id = 0; y_id = 2;
		}
	}
	else if (fabs(ds2_norm - hinterdistance) < 0.03)
	{
		if (fabs(ds1_norm - vinterdistance) < 0.03)
		{
			origo_id = 0; x_id = 2; y_id = 1;
		}
		else if (fabs(ds3_norm - vinterdistance) < 0.03)
		{
			origo_id = 2; x_id = 0; y_id = 1;
		}
	}
	else if (fabs(ds3_norm - hinterdistance) < 0.03)
	{
		if (fabs(ds1_norm - vinterdistance) < 0.03)
		{
			origo_id = 1; x_id = 2; y_id = 0;
		}
		else if (fabs(ds2_norm - vinterdistance) < 0.03)
		{
			origo_id = 2; x_id = 1; y_id = 0;
		}
	}

	if (origo_id == -1)
		return;// false

#if 1
	// do a L-M optimization
	Eigen::VectorXd optx(9);
	optx(0) = ringCircles[origo_id].t(0);
	optx(1) = ringCircles[origo_id].t(1);
	optx(2) = ringCircles[origo_id].t(2);
	optx(3) = ringCircles[x_id].t(0);
	optx(4) = ringCircles[x_id].t(1);
	optx(5) = ringCircles[x_id].t(2);
	optx(6) = ringCircles[y_id].t(0);
	optx(7) = ringCircles[y_id].t(1);
	optx(8) = ringCircles[y_id].t(2);

	std::cout << "x: " << optx << std::endl;

	marker3_functor functor(hinterdistance, vinterdistance);
	Eigen::NumericalDiff<marker3_functor> numDiff(functor);
	Eigen::LevenbergMarquardt<Eigen::NumericalDiff<marker3_functor>, double> lm(numDiff);
	lm.parameters.maxfev = 20;
	lm.parameters.xtol = 1.0e-10;
	std::cout << lm.parameters.maxfev << std::endl;
	int ret = lm.minimize(optx);

	std::cout << lm.iter << std::endl;
	std::cout << ret << std::endl;
	std::cout << "x that minimizes the function: " << optx << std::endl;

	ringCircles[origo_id].t = cv::Vec3f(optx(0), optx(1), optx(2));
	ringCircles[x_id].t = cv::Vec3f(optx(3), optx(4), optx(5));
	ringCircles[y_id].t = cv::Vec3f(optx(6), optx(7), optx(8));
#endif

	ds1 = ringCircles[x_id].t - ringCircles[origo_id].t;
	ds2 = ringCircles[y_id].t - ringCircles[origo_id].t;

	// correct
	// first, estimate r;
	cv::Vec3f ai = ds1 / cv::norm(ds1);
	cv::Vec3f aj = ds2 / cv::norm(ds2);
	cv::Vec3f ak = ai.cross(aj);

	cv::Mat r(3, 3, CV_32F);
	r.ptr<float>(0)[0] = ai(0); r.ptr<float>(0)[1] = aj(0); r.ptr<float>(0)[2] = ak(0);
	r.ptr<float>(1)[0] = ai(1); r.ptr<float>(1)[1] = aj(1); r.ptr<float>(1)[2] = ak(1);
	r.ptr<float>(2)[0] = ai(2); r.ptr<float>(2)[1] = aj(2); r.ptr<float>(2)[2] = ak(2);

	ringCircles[0].r33 = r;
	ringCircles[1].r33 = r;

	// draw axis
	cv::Vec3f origin = ringCircles[origo_id].t;
	cv::Vec3f ptx = origin + 0.1 * ai;
	cv::Vec3f pty = origin + 0.1 * aj;
	cv::Vec3f ptz = origin + 0.1 * ak;

	cv::Point2f origin2d, ptx2d, pty2d, ptz2d;
	origin2d.x = (origin(0) / origin(2))*camK.at<double>(0, 0) + camK.at<double>(0, 2);
	origin2d.y = (origin(1) / origin(2))*camK.at<double>(1, 1) + camK.at<double>(1, 2);
	ptx2d.x = (ptx(0) / ptx(2))*camK.at<double>(0, 0) + camK.at<double>(0, 2);
	ptx2d.y = (ptx(1) / ptx(2))*camK.at<double>(1, 1) + camK.at<double>(1, 2);
	pty2d.x = (pty(0) / pty(2))*camK.at<double>(0, 0) + camK.at<double>(0, 2);
	pty2d.y = (pty(1) / pty(2))*camK.at<double>(1, 1) + camK.at<double>(1, 2);
	ptz2d.x = (ptz(0) / ptz(2))*camK.at<double>(0, 0) + camK.at<double>(0, 2);
	ptz2d.y = (ptz(1) / ptz(2))*camK.at<double>(1, 1) + camK.at<double>(1, 2);

	// 3 axis
	cv::arrowedLine(src, origin2d, ptx2d, cv::Scalar(255, 0, 0, 0), 2, 8, 0);
	cv::arrowedLine(src, origin2d, pty2d, cv::Scalar(0, 255, 0, 0), 2, 8, 0);
	cv::arrowedLine(src, origin2d, ptz2d, cv::Scalar(0, 0, 255, 0), 2, 8, 0);
}

bool circularPatternBasedLocSystems::setAxisFrame(std::vector<cv::Point> &click, std::string &axisFile)
{
	std::vector<int> matchedid;
	matchedid.resize(4);
	for (size_t i = 0; i < ringCircles.size(); i++)
	{
		int x = ringCircles[i].outter.x;
		int y = ringCircles[i].outter.y;
		int d = abs(click[0].x - x) + abs(click[0].y - y);
		if (d < 30)
		{
			matchedid[0] = i;
			continue;
		}
		d = abs(click[1].x - x) + abs(click[1].y - y);
		if (d < 30)
		{
			matchedid[1] = i;
			continue;
		}
		d = abs(click[2].x - x) + abs(click[2].y - y);
		if (d < 30)
		{
			matchedid[2] = i;
			continue;
		}
		d = abs(click[3].x - x) + abs(click[3].y - y);
		if (d < 30)
		{
			matchedid[3] = i;
			continue;
		}
	}

	cv::Vec3f poses[4];
	poses[0] = ringCircles[matchedid[0]].t;
	poses[1] = ringCircles[matchedid[1]].t;
	poses[2] = ringCircles[matchedid[2]].t;
	poses[3] = ringCircles[matchedid[3]].t;

	//// set (0,0) of circle at top, left
	///*std::swap(origin_circles[zero_i], origin_circles[0]);
	//std::swap(circle_poses[zero_i], circle_poses[0]);*/
	//cv::Vec3f vecs[3];
	//for (int i = 0; i < 3; i++) {
	//	vecs[i] = poses[i + 1] - poses[0];
	//	cout << "vec " << i + 1 << "->0 " << vecs[i] << endl;
	//}
	//int min_prod_i = 0;
	//double min_prod = 1e6;
	//for (int i = 0; i < 3; i++) {
	//	float prod = fabsf(vecs[(i + 2) % 3].dot(vecs[i]));
	//	cout << "prod: " << ((i + 2) % 3 + 1) << " " << i + 1 << " " << vecs[(i + 2) % 3] << " " << vecs[i] << " " << prod << endl;
	//	if (prod < min_prod) { min_prod = prod; min_prod_i = i; }
	//}
	//int axis1_i = (((min_prod_i + 2) % 3) + 1);
	//int axis2_i = (min_prod_i + 1);
	//if (fabsf(poses[axis1_i](0)) < fabsf(poses[axis2_i](0))) std::swap(axis1_i, axis2_i);
	//int xy_i = 0;
	//for (int i = 1; i <= 3; i++) if (i != axis1_i && i != axis2_i) { xy_i = i; break; }
	//cout << "axis ids: " << axis1_i << " " << axis2_i << " " << xy_i << endl;
	
	cv::Vec3f posesReordered[4];
	posesReordered[0] = poses[0];
	posesReordered[1] = poses[1];
	posesReordered[2] = poses[2];
	posesReordered[3] = poses[3];

	float dim_y = yaxisl;//TODO
	float dim_x = xaxisl;

	cv::Vec2f targets[4] = { cv::Vec2f(0,0), cv::Vec2f(dim_x, 0), cv::Vec2f(0, dim_y), cv::Vec2f(dim_x, dim_y) };

	// build matrix of coefficients and independent term for linear eq. system
	cv::Mat A(8, 8, CV_64FC1), b(8, 1, CV_64FC1), x(8, 1, CV_64FC1);

	cv::Vec2f tmp[4];
	for (int i = 0; i < 4; i++) tmp[i] = cv::Vec2f(posesReordered[i](0), posesReordered[i](1)) / posesReordered[i](2);
	for (int i = 0; i < 4; i++) {
		cv::Mat r_even = (cv::Mat_<double>(1, 8) << -tmp[i](0), -tmp[i](1), -1, 0, 0, 0, targets[i](0) * tmp[i](0), targets[i](0) * tmp[i](1));
		cv::Mat r_odd = (cv::Mat_<double>(1, 8) << 0, 0, 0, -tmp[i](0), -tmp[i](1), -1, targets[i](1) * tmp[i](0), targets[i](1) * tmp[i](1));
		r_even.copyTo(A.row(2 * i));
		r_odd.copyTo(A.row(2 * i + 1));
		b.at<double>(2 * i) = -targets[i](0);
		b.at<double>(2 * i + 1) = -targets[i](1);
	}

	// solve linear system and obtain transformation
	cv::solve(A, b, x);
	x.push_back(1.0);
	coordinateTransform = x.reshape(1, 3);
	cout << "H " << coordinateTransform << endl;

	// TODO: compare H obtained by OpenCV with the hand approach
	std::vector<cv::Vec2f> src(4), dsts(4);
	for (int i = 0; i < 4; i++) {
		src[i] = tmp[i];
		dsts[i] = targets[i];
		cout << tmp[i] << " -> " << targets[i] << endl;
	}
	cv::Matx33f H = cv::findHomography(src, dsts, CV_LMEDS);
	cout << "OpenCV H " << H << endl;

	frame[0].x = ringCircles[matchedid[0]].outter.x; frame[0].y = ringCircles[matchedid[0]].outter.y;
	frame[1].x = ringCircles[matchedid[1]].outter.x; frame[1].y = ringCircles[matchedid[1]].outter.y;
	frame[2].x = ringCircles[matchedid[2]].outter.x; frame[2].y = ringCircles[matchedid[2]].outter.y;
	frame[3].x = ringCircles[matchedid[3]].outter.x; frame[3].y = ringCircles[matchedid[3]].outter.y;

	if (!axisFile.empty()) {
		cv::FileStorage fs(axisFile, cv::FileStorage::WRITE);
		fs << "H" << cv::Mat(cv::Matx33d(coordinateTransform)); // store as double to get more decimals
		fs << "c0" << "{" << "x" << ringCircles[matchedid[0]].outter.x << "y" << ringCircles[matchedid[0]].outter.y << "}";
		fs << "c1" << "{" << "x" << ringCircles[matchedid[1]].outter.x << "y" << ringCircles[matchedid[1]].outter.y << "}";
		fs << "c2" << "{" << "x" << ringCircles[matchedid[2]].outter.x << "y" << ringCircles[matchedid[2]].outter.y << "}";
		fs << "c3" << "{" << "x" << ringCircles[matchedid[3]].outter.x << "y" << ringCircles[matchedid[3]].outter.y << "}";
	}
	setAxis = true;
	return true;
}


void circularPatternBasedLocSystems::read_axis(const std::string& file) {
	cv::FileStorage fs(file, cv::FileStorage::READ);
	cv::Mat m;
	fs["H"] >> m;
	coordinateTransform = cv::Matx33f(m);
	cv::FileNode node = fs["c0"];
	frame[0].x = (float)node["x"];
	frame[0].y = (float)node["y"];
	node = fs["c1"];
	frame[1].x = (float)node["x"];
	frame[1].y = (float)node["y"];
	node = fs["c2"];
	frame[2].x = (float)node["x"];
	frame[2].y = (float)node["y"];
	node = fs["c3"];
	frame[3].x = (float)node["x"];
	frame[3].y = (float)node["y"];
	setAxis = true;
	cout << "transformation: " << coordinateTransform << endl;
}

void circularPatternBasedLocSystems::draw_axis(cv::Mat& image)
{
	static std::string names[4] = { "0,0", "1,0", "0,1", "1,1" };
	for (int i = 0; i < 4; i++) {
		//ostr << std::fixed << std::setprecision(5) << names[i] << endl << get_pose(origin_circles[i]).pos;
		cv::circle(image, frame[i], 1, cv::Vec3b((i == 0 || i == 3 ? 255 : 0), (i == 1 ? 255 : 0), (i == 2 || i == 3 ? 255 : 0)), -1, 8, 0);
		cv::putText(image, names[i], frame[i], cv::FONT_HERSHEY_SIMPLEX, 2, cv::Vec3b((i == 0 || i == 3 ? 255 : 0), (i == 1 ? 255 : 0), (i == 2 || i == 3 ? 255 : 0)),
			2, 8);
	}

	cv::arrowedLine(image, cv::Point(frame[0].x, frame[0].y), cv::Point(frame[1].x, frame[1].y),
		cv::Scalar(0, 0, 255, 0), 2, 8, 0);
	cv::arrowedLine(image, cv::Point(frame[0].x, frame[0].y), cv::Point(frame[2].x, frame[2].y),
		cv::Scalar(255, 0, 0, 0), 2, 8, 0);
	cv::arrowedLine(image, cv::Point(frame[0].x, frame[0].y), cv::Point(frame[3].x, frame[3].y),
		cv::Scalar(0, 255, 0, 0), 2, 8, 0);
}


void circularPatternBasedLocSystems::drawPatterns(cv::Mat frame_rgb)
{
	for (size_t i = 0; i < ringCircles.size(); i++)
	{
		cv::RotatedRect eo;
		eo.center.x		= ringCircles[i].outter.x;
		eo.center.y		= ringCircles[i].outter.y;
		eo.size.width	= ringCircles[i].outter.a*2;
		eo.size.height	= ringCircles[i].outter.b*2;
		eo.angle		= ringCircles[i].outter.theta;

		cv::RotatedRect ei;
		ei.center.x = ringCircles[i].inner.x;
		ei.center.y = ringCircles[i].inner.y;
		ei.size.width = ringCircles[i].inner.a*2;
		ei.size.height = ringCircles[i].inner.b*2;
		ei.angle = ringCircles[i].inner.theta;

		cv::circle(frame_rgb, eo.center, 2, cv::Scalar(0, 255, 0, 0), -1, 8);
		cv::circle(frame_rgb, ei.center, 2, cv::Scalar(0, 255, 255, 0), -1, 8);

		float sx1, sx2, sy1, sy2;
		//endpoints in image coords
		sx1 = ringCircles[i].outter.x + ringCircles[i].outter.v0 * ringCircles[i].outter.a;
		sx2 = ringCircles[i].outter.x - ringCircles[i].outter.v0 * ringCircles[i].outter.a;
		sy1 = ringCircles[i].outter.y + ringCircles[i].outter.v1 * ringCircles[i].outter.a;
		sy2 = ringCircles[i].outter.y - ringCircles[i].outter.v1 * ringCircles[i].outter.a;
		cv::line(frame_rgb, cv::Point(sx1, sy1), cv::Point(sx2, sy2), cv::Scalar(0, 0, 255, 0), 2, 8);

		//calculate the minor axis 
		//endpoints in image coords
		sx1 = ringCircles[i].outter.x + ringCircles[i].outter.v1 * ringCircles[i].outter.b;
		sx2 = ringCircles[i].outter.x - ringCircles[i].outter.v1 * ringCircles[i].outter.b;
		sy1 = ringCircles[i].outter.y - ringCircles[i].outter.v0 * ringCircles[i].outter.b;
		sy2 = ringCircles[i].outter.y + ringCircles[i].outter.v0 *  ringCircles[i].outter.b;
		cv::line(frame_rgb, cv::Point(sx1, sy1), cv::Point(sx2, sy2), cv::Scalar(255, 0, 0, 0), 2, 8);

		std::ostringstream ss;

#define DISPLAY_COOR(t) \
	{\
		float x = static_cast<int>(t(0) * 1000)/1000.0;\
		float y = static_cast<int>(t(1) * 1000) / 1000.0;\
		float z = static_cast<int>(t(2) * 1000) / 1000.0;\
		ss << std::setprecision(3) << "[" << x << "," << y << ","<< z << "]";\
	}
		DISPLAY_COOR(ringCircles[i].t)
		cv::putText(frame_rgb, ss.str(), eo.center, CV_FONT_HERSHEY_COMPLEX_SMALL, 2, cv::Scalar(0, 0, 255, 0), 2, 8);
		//ss.clear();
		//cv::ellipse(frame_rgb, eo, cv::Scalar(0, 0, 255, 0), 2, 8);
		//cv::ellipse(frame_rgb, ei, cv::Scalar(255, 0, 0, 0), 2, 8);
		//cv::drawContours(frame_rgb, contours, ringCircles[i].matchpair.first, cv::Scalar(0, 0, 255, 0), 1, 8);
		//cv::drawContours(frame, contours, ringCircles[i].matchpair.second, cv::Scalar(255, 0, 0, 0), 1, 8);
	}
}

bool circularPatternBasedLocSystems::tostream(std::ofstream &oss)
{
	if (ringCircles.size() == 0)
	{
		oss << -1 << "," << -1 << "," << -1 << "," << -1 << ";";
		oss << std::endl;
		return false;
	}

	for (size_t i = 0; i < ringCircles.size(); i++)
	{
		std::ostringstream ss;

#define DISPLAY_COOR(t) \
	{\
		float x = static_cast<int>(t(0) * 1000)/1000.0;\
		float y = static_cast<int>(t(1) * 1000) / 1000.0;\
		float z = static_cast<int>(t(2) * 1000) / 1000.0;\
		ss << std::setprecision(5) << x << "," << y << "," << z;\
	}

		DISPLAY_COOR(ringCircles[i].t)
			oss << i << "," << ss.str() << ";";
	}
	oss << std::endl;

	return true;
}
