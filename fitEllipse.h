#ifndef __FIE_ELLIPSE_H__
#define __FIE_ELLIPSE_H__

#include "opencv2/core/core.hpp"
#include "opencv2/core/mat.hpp"

cv::RotatedRect fitEllipseDirect(cv::InputArray _points, cv::Mat &rawSol);

void fromRotatedRectToEllipseParams(const cv::RotatedRect &rect, cv::Mat &et);

cv::Vec3f estimatePositionAnalyticalSol(const cv::Mat &et, const cv::Mat &camMatrix, float diameter);

cv::Vec3f estimatePositionGeometricSol(const cv::RotatedRect &rect, const cv::Mat &camMatrix, float diameter);

#endif