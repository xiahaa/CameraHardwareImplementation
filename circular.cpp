#include <cstdio>
#include <cmath>
#include "circular.h"
#include <limits>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <limits>
#include <math.h>


using namespace std;

using std::cout;
using std::endl;
using std::numeric_limits;
#ifndef M_PI
#define M_PI       3.14159265358979323846   // pi
#endif

#define min(a,b) ((a) < (b) ? (a) : (b))
#define max(a,b) ((a) > (b) ? (a) : (b))

#define MAX_SEGMENTS 10000 // TODO: necessary?
#define CIRCULARITY_TOLERANCE 0.02

#ifndef M_PI
#define M_PI       3.14159265358979323846   // pi
#endif

#pragma warning(disable : 4305)  
#pragma warning(disable : 4244)  

cv::CircleDetector::CircleDetector(int _width, int _height, Context* _context, float _diameter_ratio)
	: context(_context)
{
	minSize = 10;
	maxSize = 100 * 100; // TODO: test!
	centerDistanceToleranceRatio = 0.1;
	centerDistanceToleranceAbs = 5;
	circularTolerance = 0.3;
	ratioTolerance = 1.0;

	//initialization - fixed params
	width = _width;
	height = _height;
	len = width*height;
	siz = len * 3;
	diameterRatio = _diameter_ratio;
	float areaRatioInner_Outer = diameterRatio*diameterRatio;
	outerAreaRatio = M_PI*(1.0 - areaRatioInner_Outer) / 4;
	innerAreaRatio = M_PI / 4.0;
	areasRatio = (1.0 - areaRatioInner_Outer) / areaRatioInner_Outer;
	// << "outerRatio/innerRatio " << outerAreaRatio << " " << innerAreaRatio << " " << diameterRatio << endl;

	threshold = (2 * 256) / 3;
	threshold_counter = 0;
}

cv::CircleDetector::~CircleDetector()
{
}

int cv::CircleDetector::get_threshold(void) const
{
	return threshold;
}

void cv::CircleDetector::change_threshold(void)
{
#if !(ENABLE_RANDOMIZED_THRESHOLD)
	threshold_counter++;
	int d = threshold_counter;
	int div = 1;
	while (d > 1) {
		d /= 2;
		div *= 2;
	}
	int step = 256 / div;
	threshold = 3 * (step * (threshold_counter - div) + step / 2);
	if (step <= 16) threshold_counter = 0;
#else
	threshold = (rand() % 48) * 16;
#endif
	cout << "attempting new threshold: " << threshold << endl;
}

bool cv::CircleDetector::examineCircle(const cv::Mat& image, cv::CircleDetector::Circle& circle, int ii, float areaRatio)
{
	//int64_t ticks = cv::getTickCount();  
	// get shorter names for elements in Context
	vector<int>& buffer = context->buffer;
	vector<int>& queue = context->queue;

	int vx, vy;
	queueOldStart = queueStart;
	int position = 0;
	int pos;
	bool result = false;
	int type = buffer[ii];
	int maxx, maxy, minx, miny;
	int pixel_class;

	//cout << "examine (type " << type << ") at " << ii / width << "," << ii % width << " (numseg " << numSegments << ")" << endl;

	buffer[ii] = ++numSegments;
	circle.x = ii%width;
	circle.y = ii / width;
	minx = maxx = circle.x;
	miny = maxy = circle.y;
	circle.valid = false;
	circle.round = false;
	//push segment coords to the queue
	queue[queueEnd++] = ii;
	//and until queue is empty
	while (queueEnd > queueStart) {
		//pull the coord from the queue
		position = queue[queueStart++];
		//search neighbours

		pos = position + 1;
		pixel_class = buffer[pos];
		if (pixel_class == 0) {
			uchar* ptr = &image.data[pos];
			pixel_class = ((ptr[0]) > threshold) - 2;
			if (pixel_class != type) buffer[pos] = pixel_class;
		}
		if (pixel_class == type) {
			queue[queueEnd++] = pos;
			maxx = max(maxx, pos%width);
			buffer[pos] = numSegments;
		}

		pos = position - 1;
		pixel_class = buffer[pos];
		if (pixel_class == 0) {
			uchar* ptr = &image.data[pos];
			pixel_class = ((ptr[0]) > threshold) - 2;
			if (pixel_class != type) buffer[pos] = pixel_class;
		}
		if (pixel_class == type) {
			queue[queueEnd++] = pos;
			minx = min(minx, pos%width);
			buffer[pos] = numSegments;
		}

		pos = position - width;
		pixel_class = buffer[pos];
		if (pixel_class == 0) {
			uchar* ptr = &image.data[pos];
			pixel_class = ((ptr[0]) > threshold) - 2;
			if (pixel_class != type) buffer[pos] = pixel_class;
		}
		if (pixel_class == type) {
			queue[queueEnd++] = pos;
			miny = min(miny, pos / width);
			buffer[pos] = numSegments;
		}

		pos = position + width;
		pixel_class = buffer[pos];
		if (pixel_class == 0) {
			uchar* ptr = &image.data[pos];
			pixel_class = ((ptr[0]) > threshold) - 2;
			if (pixel_class != type) buffer[pos] = pixel_class;
		}
		if (pixel_class == type) {
			queue[queueEnd++] = pos;
			maxy = max(maxy, pos / width);
			buffer[pos] = numSegments;
		}

		//if (queueEnd-queueOldStart > maxSize) return false;
	}

	//once the queue is empty, i.e. segment is complete, we compute its size 
	circle.size = queueEnd - queueOldStart;
	if (circle.size > minSize) {
		//and if its large enough, we compute its other properties 
		circle.maxx = maxx;
		circle.maxy = maxy;
		circle.minx = minx;
		circle.miny = miny;
		circle.type = -type;
		vx = (circle.maxx - circle.minx + 1);
		vy = (circle.maxy - circle.miny + 1);
		circle.x = (circle.maxx + circle.minx) / 2;
		circle.y = (circle.maxy + circle.miny) / 2;
		circle.roundness = vx*vy*areaRatio / circle.size;
		//we check if the segment is likely to be a ring 
		if (circle.roundness - circularTolerance < 1.0 && circle.roundness + circularTolerance > 1.0)
		{
			//if its round, we compute yet another properties 
			circle.round = true;

			// TODO: mean computation could be delayed until the inner ring also satisfies above condition, right?
			circle.mean = 0;
			for (int p = queueOldStart; p<queueEnd; p++) {
				pos = queue[p];
				circle.mean += image.data[pos];
			}
			circle.mean = circle.mean / circle.size;
			result = true;
			// << "segment size " << circle.size << " " << vx << " " << vy << endl;
		}
		// else cout << "not round enough (" << circle.roundness << ") vx/vy " << vx << " " << vy 
		// << " ctr " << circle.x << " " << circle.y << " " << circle.size << " " << areaRatio << endl;
	}

	//double delta = (double)(cv::getTickCount() - ticks) / cv::getTickFrequency();
	//cout << "examineCircle: " << delta << " " << " fps: " << 1/delta << " pix: " << circle.size << " " << threshold << endl;

	return result;
}

cv::CircleDetector::Circle cv::CircleDetector::detect(const cv::Mat& image, const cv::CircleDetector::Circle& previous_circle)
{
	vector<int>& buffer = context->buffer;
	vector<int>& queue = context->queue;

	int pos = (height - 1)*width;
	int ii = 0;
	int start = 0;
	Circle inner, outer;
	//int outer_id;

	if (previous_circle.valid) {
		ii = ((int)previous_circle.y)*width + (int)previous_circle.x;
		start = ii;
	}

	//cout << "detecting (thres " << threshold << ") at " << ii << endl;

	numSegments = 0;

	const int nchannels = 1;

	do
	{
		if (numSegments > MAX_SEGMENTS) break;
		//if (start != 0) cout << "it " << ii << endl;

		// if current position needs to be thresholded
		int pixel_class = buffer[ii];
		if (pixel_class == 0) {
			uchar* ptr = &image.data[ii * nchannels];
			//cout << "value: " << (ptr[0]+ptr[1]+ptr[2]) << endl;
			pixel_class = (ptr[0] > threshold) - 2;

			if (pixel_class == -2) buffer[ii] = pixel_class;
			// only tag black pixels, to avoid dirtying the buffer outside the ellipse
			// NOTE: the inner white area will not initially be tagged, but once the inner circle is processed, it will
		}

		//cout << pixel_class << endl;

		// if the current pixel is detected as "black"
		if (pixel_class == -2) {
			queueEnd = 0;
			queueStart = 0;
			// << "black pixel " << ii << endl;

			// check if looks like the outer portion of the ring
			if (examineCircle(image, outer, ii, outerAreaRatio)) {
				pos = outer.y * width + outer.x; // jump to the middle of the ring

												 // treshold the middle of the ring and check if it is detected as "white"
				pixel_class = buffer[pos];
				if (pixel_class == 0) {
					uchar* ptr = &image.data[pos * nchannels];
					pixel_class = ((ptr[0]) > threshold) - 2;
					buffer[pos] = pixel_class;
				}
				if (pixel_class == -1) {

					// check if it looks like the inner portion
					if (examineCircle(image, inner, pos, innerAreaRatio)) {
						// it does, now actually check specific properties to see if it is a valid target
						if (
							((float)outer.size / areasRatio / (float)inner.size - ratioTolerance < 1.0 &&
							(float)outer.size / areasRatio / (float)inner.size + ratioTolerance > 1.0) &&
								(fabsf(inner.x - outer.x) <= centerDistanceToleranceAbs + centerDistanceToleranceRatio * ((float)(outer.maxx - outer.minx))) &&
							(fabsf(inner.y - outer.y) <= centerDistanceToleranceAbs + centerDistanceToleranceRatio * ((float)(outer.maxy - outer.miny)))
							)
						{
							float cm0, cm1, cm2;
							cm0 = cm1 = cm2 = 0;
							inner.x = outer.x;
							inner.y = outer.y;

							// computer centroid
							float sx = 0;
							float sy = 0;
							queueOldStart = 0;
							for (int p = 0; p<queueEnd; p++) {
								pos = queue[p];
								sx += pos % width;
								sy += pos / width;
							}
							// update pixel-based position oreviously computed
							inner.x = sx / queueEnd;
							inner.y = sy / queueEnd;
							outer.x = sx / queueEnd;
							outer.y = sy / queueEnd;

							// compute covariance
							for (int p = 0; p<queueEnd; p++) {
								pos = queue[p];
								float tx = pos % width - outer.x;
								float ty = pos / width - outer.y;
								cm0 += tx * tx;
								cm2 += ty * ty;
								cm1 += tx * ty;
							}

							float fm0, fm1, fm2;
							fm0 = ((float)cm0) / queueEnd; // cov(x,x)
							fm1 = ((float)cm1) / queueEnd; // cov(x,y)
							fm2 = ((float)cm2) / queueEnd; // cov(y,y)

							float trace = fm0 + fm2; // sum of elements in diag.
							float det = trace * trace - 4 * (fm0 * fm2 - fm1 * fm1);
							if (det > 0) det = sqrt(det); else det = 0;                    //yes, that can happen as well:(
							float f0 = (trace + det) / 2;
							float f1 = (trace - det) / 2;
							inner.m0 = sqrt(f0);
							inner.m1 = sqrt(f1);
							if (fm1 != 0) {                               //aligned ?
								inner.v0 = -fm1 / sqrt(fm1*fm1 + (fm0 - f0)*(fm0 - f0)); // no-> standard calculatiion
								inner.v1 = (fm0 - f0) / sqrt(fm1*fm1 + (fm0 - f0)*(fm0 - f0));
							}
							else {
								inner.v0 = inner.v1 = 0;
								if (fm0 > fm2) inner.v0 = 1.0; else inner.v1 = 1.0;   // aligned, hm, is is aligned with x or y ?
							}

							inner.bwRatio = (float)outer.size / inner.size;

							// TODO: purpose? should be removed? if next if fails, it will go over and over to the same place until number of segments
							// reaches max, right?
							ii = start - 1; // track position 

							float circularity = M_PI * 4 * (inner.m0)*(inner.m1) / queueEnd;
							if (fabsf(circularity - 1) < CIRCULARITY_TOLERANCE) {
								outer.valid = inner.valid = true; // at this point, the target is considered valid
																  /*inner_id = numSegments; outer_id = numSegments - 1;*/
								threshold = (outer.mean + inner.mean) / 2; // use a new threshold estimate based on current detection
								cout << "threshold set to: " << threshold << endl;

								//pixel leakage correction
								float r = diameterRatio*diameterRatio;
								float m0o = sqrt(f0);
								float m1o = sqrt(f1);
								float ratio = (float)inner.size / (outer.size + inner.size);
								float m0i = sqrt(ratio)*m0o;
								float m1i = sqrt(ratio)*m1o;
								float a = (1 - r);
								float b = -(m0i + m1i) - (m0o + m1o)*r;
								float c = (m0i*m1i) - (m0o*m1o)*r;
								float t = (-b - sqrt(b*b - 4 * a*c)) / (2 * a);
								m0i -= t; m1i -= t; m0o += t; m1o += t;

								inner.m0 = sqrt(f0) + t;
								inner.m1 = sqrt(f1) + t;
								inner.minx = outer.minx;
								inner.maxx = outer.maxx;
								inner.maxy = outer.maxy;
								inner.miny = outer.miny;
								break;
							}
						}
					}
				}
			}
		}
		ii++;
		if (ii >= len) ii = 0;
	} while (ii != start);

	// draw
	if (inner.valid)
		threshold_counter = 0;
	else
		change_threshold();
	// update threshold for next run. inner is what user receives

	//cv::namedWindow("buffer", CV_WINDOW_NORMAL);
	//cv::Mat buffer_img;
	//context->debug_buffer(image, buffer_img);
	//cv::imshow("buffer", buffer_img);
	//cv::waitKey(100);

	// if this is not the first call (there was a previous valid circle where search started),
	// the current call found a valid match, and only two segments were found during the search (inner/outer)
	// then, only the bounding box area of the outer ellipse needs to be cleaned in 'buffer'
	bool fast_cleanup = (previous_circle.valid && numSegments == 2 && inner.valid);
	context->cleanup(outer, fast_cleanup);

	if (!inner.valid) cout << "detection failed" << endl;
	else cout << "detected at " << inner.x << " " << inner.y << endl;

	return inner;
}


void cv::CircleDetector::cover_last_detected(cv::Mat& image)
{
	const vector<int>& queue = context->queue;
	for (int i = queueOldStart; i < queueEnd; i++) {
		int pos = queue[i];
		uchar* ptr = image.data + pos;
		*ptr = 255;
	}
}

void cv::CircleDetector::improveEllipse(const cv::Mat& image, Circle& c)
{
	cv::Mat subimg;
	int delta = 10;
	cout << image.rows << " x " << image.cols << endl;
	cv::Range row_range(max(0, c.miny - delta), min(c.maxy + delta, image.rows));
	cv::Range col_range(max(0, c.minx - delta), min(c.maxx + delta, image.cols));
	cout << row_range.start << " " << row_range.end << " " << col_range.start << " " << col_range.end << endl;
	image(row_range, col_range).copyTo(subimg);
	cv::Mat cannified;
	cv::Canny(subimg, cannified, 4000, 8000, 5, true);

	/*cv::namedWindow("bleh");
	cv::imshow("bleh", subimg);
	cv::waitKey();*/

	std::vector< std::vector<cv::Point> > contours;
	cv::findContours(cannified, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	if (contours.empty() || contours[0].size() < 5) return;

	cv::Mat contour_img;
	subimg.copyTo(contour_img);
	cv::drawContours(contour_img, contours, 0, cv::Scalar(255, 0, 255), 1);

	/*cv::namedWindow("bleh2");
	cv::imshow("bleh2", contour_img);
	cv::waitKey();*/


	cv::RotatedRect rect = cv::fitEllipse(contours[0]);
	cout << "old: " << c.x << " " << c.y << " " << c.m0 << " " << c.m1 << " " << c.v0 << " " << c.v1 << endl;
	c.x = rect.center.x + col_range.start;
	c.y = rect.center.y + row_range.start;
	/*float max_size = max(rect.size.width, rect.size.height);
	float min_size = min(rect.size.width, rect.size.height);*/
	c.m0 = rect.size.width * 0.25;
	c.m1 = rect.size.height * 0.25;
	c.v0 = cos(rect.angle / 180.0 * M_PI);
	c.v1 = sin(rect.angle / 180.0 * M_PI);
	cout << "new: " << c.x << " " << c.y << " " << c.m0 << " " << c.m1 << " " << c.v0 << " " << c.v1 << endl;

	/*cv::Mat ellipse_img;
	image(row_range, col_range).copyTo(subimg);
	subimg.copyTo(ellipse_img);
	cv::ellipse(ellipse_img, rect, cv::Scalar(255,0,255));
	cv::namedWindow("bleh3");
	cv::imshow("bleh3", ellipse_img);
	cv::waitKey();*/
}


cv::CircleDetector::Circle::Circle(void)
{
	x = y = 0;
	round = valid = false;
}

cv::Point2i cv::CircleDetector::Circle::returnEllipseCenter() const
{
	return cv::Point2i(x + 2 * m0, y + 2 * m1);
}

void cv::CircleDetector::Circle::draw(cv::Mat& image, const std::string& text, cv::Vec3b color, float thickness) const
{
	//cv::circle(image, cv::Point(x, y), 1, color, 1, 4);
	//cv::ellipse(image, cv::Point(x, y), cv::Size2f(m0 * 2, m1 * 2), atan2(v1, v0)  * 180.0 / M_PI, 0, 360, color, thickness, 8);

	for (float e = 0; e < 2 * M_PI; e += 0.05) {
		float fx = x + cos(e) * v0 * m0 * 2 + v1 * m1 * 2 * sin(e);
		float fy = y + cos(e) * v1 * m0 * 2 - v0 * m1 * 2 * sin(e);
		int fxi = (int)(fx + 0.5);
		int fyi = (int)(fy + 0.5);
		if (fxi >= 0 && fxi < image.cols && fyi >= 0 && fyi < image.rows)
			image.at<cv::Vec3b>(fyi, fxi) = color;
	}

	float scale = image.size().width / 1800.0f * 0.7;
	//float thickness = scale * 3.0;
	//if (thickness < 1) thickness = 1;
	cv::putText(image, text.c_str(), cv::Point(x + 2 * m0 - 15, y + 2 * m1 + 15), CV_FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(color), thickness, CV_AA);
	cv::line(image, cv::Point(x + v0 * m0 * 2, y + v1 * m0 * 2), cv::Point(x - v0 * m0 * 2, y - v1 * m0 * 2), cv::Scalar(color), 1, 8);
	cv::line(image, cv::Point(x + v1 * m1 * 2, y - v0 * m1 * 2), cv::Point(x - v1 * m1 * 2, y + v0 * m1 * 2), cv::Scalar(color), 1, 8);
}

void cv::CircleDetector::Circle::write(cv::FileStorage& fs) const {
	fs << "{" << "x" << x << "y" << y << "size" << size <<
		"maxy" << maxy << "maxx" << maxx << "miny" << miny << "minx" << minx <<
		"mean" << mean << "type" << type << "roundness" << roundness << "bwRatio" << bwRatio <<
		"round" << round << "valid" << valid << "m0" << m0 << "m1" << m1 << "v0" << v0 << "v1" << v1 << "}";
}

void cv::CircleDetector::Circle::read(const cv::FileNode& node)
{
	x = (float)node["x"];
	y = (float)node["y"];
	size = (int)node["size"];
	maxy = (int)node["maxy"];
	maxx = (int)node["maxx"];
	miny = (int)node["miny"];
	minx = (int)node["minx"];
	mean = (int)node["mean"];
	type = (int)node["type"];
	roundness = (float)node["roundness"];
	bwRatio = (float)node["bwRatio"];
	round = (int)node["round"];
	valid = (int)node["valid"];
	m0 = (float)node["m0"];
	m1 = (float)node["m1"];
	v0 = (float)node["v0"];
	v1 = (float)node["v1"];
}

cv::CircleDetector::Context::Context(int _width, int _height)
{
	width = _width;
	height = _height;
	int len = width * height;
	buffer.resize(len);
	queue.resize(len);
	cleanup(Circle(), false);
}

void cv::CircleDetector::Context::cleanup(const Circle& c, bool fast_cleanup) {
	if (c.valid && fast_cleanup)
	{
		// zero only parts modified when detecting 'c'
		int ix = max(c.minx - 2, 1);
		int ax = min(c.maxx + 2, width - 2);
		int iy = max(c.miny - 2, 1);
		int ay = min(c.maxy + 2, height - 2);
		for (int y = iy; y < ay; y++) {
			int pos = y * width;
			for (int x = ix; x < ax; x++) buffer[pos + x] = 0; // TODO: user ptr and/or memset
		}
	}
	else {
		cout << "clean whole buffer" << endl;
		memset(&buffer[0], 0, sizeof(int)*buffer.size());

		//image delimitation
		for (int i = 0; i<width; i++) {
			buffer[i] = -1000;
			buffer[(height - 1) * width + i] = -1000;
		}
		for (int i = 0; i<height; i++) {
			buffer[width * i] = -1000;
			buffer[width * i + width - 1] = -1000;
		}
	}
}

void cv::CircleDetector::Context::debug_buffer(const cv::Mat& image, cv::Mat& out)
{
	out.create(height, width, CV_8UC3);
	cv::Vec3b* out_ptr = out.ptr<cv::Vec3b>(0);
	const cv::Vec3b* im_ptr = image.ptr<cv::Vec3b>(0);
	out = cv::Scalar(128, 128, 128);
	for (uint i = 0; i < out.total(); i++, ++out_ptr, ++im_ptr) {
		/*if (buffer[i] == -1) *ptr = cv::Vec3b(0,0,0);
		else if (buffer[i] == -2) *ptr = cv::Vec3b(255,255,255);*/
		//else if (buffer[i] < 0) *ptr = cv::Vec3b(0, 255, 0);
		if (buffer[i] > 0) *out_ptr = cv::Vec3b(255, 0, 255);
		else *out_ptr = *im_ptr;
	}
}

cv::ManyCircleDetector::ManyCircleDetector(int _number_of_circles, int _width, int _height, float _diameter_ratio) :
	context(_width, _height), width(_width), height(_height), number_of_circles(_number_of_circles)
{
	circles.resize(number_of_circles);
	detectors.resize(number_of_circles, CircleDetector(width, height, &context, _diameter_ratio));
}

cv::ManyCircleDetector::~ManyCircleDetector(void) {
}

bool cv::ManyCircleDetector::detect(const cv::Mat& image, cv::Mat &cimg, bool reset, int max_attempts, int refine_max_step) {
	bool all_detected = true;
	cv::Mat input;
	if (reset) image.copyTo(input); // image will be modified on reset
	else input = image;

	cv::Mat gray;
	if (input.channels() == 3)
		cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
	else
		input.copyTo(gray);

	for (int i = 0; i < number_of_circles && all_detected; i++)
	{
		cout << "detecting circle " << i << endl;

		for (int j = 0; j < max_attempts; j++)
		{
			cout << "attempt " << j << endl;

			for (int refine_counter = 0; refine_counter < refine_max_step; refine_counter++)
			{
				if (refine_counter > 0) cout << "refining step " << refine_counter << "/" << refine_max_step << endl;
				int prev_threshold = detectors[i].get_threshold();

				int64_t ticks = cv::getTickCount();

				if (refine_counter == 0 && reset)
					circles[i] = detectors[i].detect(gray, (i == 0 ? CircleDetector::Circle() : circles[i - 1]));
				else
					circles[i] = detectors[i].detect(gray, circles[i]);

				double delta = (double)(cv::getTickCount() - ticks) / cv::getTickFrequency();
				cout << "t: " << delta << " " << " fps: " << 1 / delta << endl;

				if (!circles[i].valid) break;

				// if the threshold changed, keep refining this circle
				if (detectors[i].get_threshold() == prev_threshold) break;
			}

			if (circles[i].valid) {
				cout << "detection of circle " << i << " ok" << endl;
				if (reset) detectors[i].cover_last_detected(gray);
				break; // detection was successful, dont keep trying
			}
		}

		if (circles[i].valid)
		{
			std::stringstream ss;
			ss << std::to_string(i);
			circles[i].draw(cimg, ss.str(), cv::Vec3b(0, 0, 255), 1);
		}
		cv::namedWindow("temp", cv::WINDOW_NORMAL);
		cv::moveWindow("temp", 200, 200);
		cv::resizeWindow("temp", 640, 480);
		cv::imshow("temp", cimg);
		cv::waitKey(0);

		// detection was not possible for this circle, abort search
		if (!circles[i].valid) { all_detected = false; break; }
	}

	return all_detected;
}


/*
localization service
*/
cv::LocalizationSystem::LocalizationSystem(
	int _targets, int _width, int _height,
	const cv::Mat& _K, const cv::Mat& _dist_coeff,
	float _outer_diameter, float _inner_diameter) :
	xscale(1), yscale(1), detector(_targets, _width, _height, _inner_diameter / _outer_diameter),
	targets(_targets), width(_width), height(_height), axis_set(false), circle_diameter(_outer_diameter)
{
	_K.copyTo(K);
	_dist_coeff.copyTo(dist_coeff);

	fc[0] = K.at<double>(0, 0);
	fc[1] = K.at<double>(1, 1);
	cc[0] = K.at<double>(0, 2);
	cc[1] = K.at<double>(1, 2);

	cout.precision(30);
	cout << "fc " << fc[0] << " " << fc[1] << endl;
	cout << "cc " << cc[0] << " " << cc[1] << endl;
	kc[0] = 1;
	cout << "kc " << kc[0] << " ";
	for (int i = 0; i < 5; i++) {
		kc[i + 1] = dist_coeff.at<double>(i);
		cout << kc[i + 1] << " ";
	}
	cout << endl;

	coordinates_transform = cv::Matx33f(1, 0, 0, 0, 1, 0, 0, 0, 1);

	precompute_undistort_map();
}

bool cv::LocalizationSystem::localize(const cv::Mat& image, bool reset, int attempts, int max_refine) {
	cv::Mat cimg;
	if (image.channels() == 3)
	{
		image.copyTo(cimg);
	}
	else
	{
		cvtColor(image, cimg, COLOR_GRAY2BGR);
	}
	return detector.detect(image, cimg, reset, attempts, max_refine);
	//return false;
}

cv::LocalizationSystem::Pose cv::LocalizationSystem::get_pose(const cv::CircleDetector::Circle& circle) const
{
	Pose result;
	float x, y, x1, x2, y1, y2, sx1, sx2, sy1, sy2, major, minor, v0, v1;

	//transform the center
	transform(circle.x, circle.y, x, y);

	//calculate the major axis 
	//endpoints in image coords
	sx1 = circle.x + circle.v0 * circle.m0 * 2;
	sx2 = circle.x - circle.v0 * circle.m0 * 2;
	sy1 = circle.y + circle.v1 * circle.m0 * 2;
	sy2 = circle.y - circle.v1 * circle.m0 * 2;

	//endpoints in camera coords 
	transform(sx1, sy1, x1, y1);
	transform(sx2, sy2, x2, y2);

	//semiaxis length 
	major = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2)) / 2.0;

	v0 = (x2 - x1) / major / 2.0;
	v1 = (y2 - y1) / major / 2.0;

	//calculate the minor axis 
	//endpoints in image coords
	sx1 = circle.x + circle.v1 * circle.m1 * 2;
	sx2 = circle.x - circle.v1 * circle.m1 * 2;
	sy1 = circle.y - circle.v0 * circle.m1 * 2;
	sy2 = circle.y + circle.v0 * circle.m1 * 2;

	//endpoints in camera coords 
	transform(sx1, sy1, x1, y1);
	transform(sx2, sy2, x2, y2);

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

#if USE_BLAS	
	//double data[] = { a,b,d,b,c,e,d,e,f };
	//result.pos = eigen(data);
#else
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
	double z = circle_diameter / sqrt(-L2*L3) / 2.0;
	cv::Matx13d position_mat = L3 * sqrt((L2 - L1) / (L2 - L3)) * eigenvectors.row(V2)
		+ L2 * sqrt((L1 - L3) / (L2 - L3)) * eigenvectors.row(V3);
	result.pos = cv::Vec3f(position_mat(0), position_mat(1), position_mat(2));
	int S3 = (result.pos(2) * z < 0 ? -1 : 1);
	result.pos *= S3 * z;
#endif

	result.rot(0) = acos(circle.m1 / circle.m0) / M_PI*180.0;
	result.rot(1) = atan2(circle.v1, circle.v0) / M_PI*180.0;
	result.rot(2) = circle.v1 / circle.v0;

	return result;
}

const cv::CircleDetector::Circle& cv::LocalizationSystem::get_circle(int id)
{
	return detector.circles[id];
}

cv::LocalizationSystem::Pose cv::LocalizationSystem::get_pose(int id) const
{
	return get_pose(detector.circles[id]);
}

cv::LocalizationSystem::Pose cv::LocalizationSystem::get_transformed_pose(int id) const {
	return get_transformed_pose(detector.circles[id]);
}

cv::LocalizationSystem::Pose cv::LocalizationSystem::get_transformed_pose(const cv::CircleDetector::Circle& circle) const
{
	Pose pose;
	pose.pos = coordinates_transform * get_pose(circle).pos;
	pose.pos(0) /= pose.pos(2);
	pose.pos(1) /= pose.pos(2);
	pose.pos(2) = 0;
	return pose;
}

// TODO: allow user to choose calibration circles, now the circles are read in the order of detection
bool cv::LocalizationSystem::set_axis(const cv::Mat& image, int max_attempts, int refine_steps, const std::string& file)
{
	ManyCircleDetector axis_detector(4, width, height);
	cv::Mat cimg;
	if (image.channels() == 3)
		image.copyTo(cimg);
	else
		cv::cvtColor(image, cimg, COLOR_GRAY2BGR);
	if (!axis_detector.detect(image, cimg, true, max_attempts, refine_steps)) return false;

	// get poses of each calibration circle
	/*float minx, miny;
	minx = miny = numeric_limits<float>::max();
	int zero_i;*/

	Pose circle_poses[4];
	for (int i = 0; i < 4; i++) {
		origin_circles[i] = axis_detector.circles[i];
		circle_poses[i] = get_pose(axis_detector.circles[i]);
		/*float x = circle_poses[i].pos(0);
		float y = circle_poses[i].pos(1);
		if (x < minx) { zero_i = i; x = minx; }
		if (y < miny) { zero_i = i; y = miny; }*/
	}

	// set (0,0) of circle at top, left
	/*std::swap(origin_circles[zero_i], origin_circles[0]);
	std::swap(circle_poses[zero_i], circle_poses[0]);*/
	cv::Vec3f vecs[3];
	for (int i = 0; i < 3; i++) {
		vecs[i] = circle_poses[i + 1].pos - circle_poses[0].pos;
		cout << "vec " << i + 1 << "->0 " << vecs[i] << endl;
	}
	int min_prod_i = 0;
	double min_prod = 1e6;
	for (int i = 0; i < 3; i++) {
		float prod = fabsf(vecs[(i + 2) % 3].dot(vecs[i]));
		cout << "prod: " << ((i + 2) % 3 + 1) << " " << i + 1 << " " << vecs[(i + 2) % 3] << " " << vecs[i] << " " << prod << endl;
		if (prod < min_prod) { min_prod = prod; min_prod_i = i; }
	}
	int axis1_i = (((min_prod_i + 2) % 3) + 1);
	int axis2_i = (min_prod_i + 1);
	if (fabsf(circle_poses[axis1_i].pos(0)) < fabsf(circle_poses[axis2_i].pos(0))) std::swap(axis1_i, axis2_i);
	int xy_i = 0;
	for (int i = 1; i <= 3; i++) if (i != axis1_i && i != axis2_i) { xy_i = i; break; }
	cout << "axis ids: " << axis1_i << " " << axis2_i << " " << xy_i << endl;

	CircleDetector::Circle origin_circles_reordered[4];
	origin_circles_reordered[0] = origin_circles[0];
	origin_circles_reordered[1] = origin_circles[axis1_i];
	origin_circles_reordered[2] = origin_circles[axis2_i];
	origin_circles_reordered[3] = origin_circles[xy_i];
	for (int i = 0; i < 4; i++) {
		origin_circles[i] = origin_circles_reordered[i];
		circle_poses[i] = get_pose(origin_circles[i]);
		cout << "original poses: " << circle_poses[i].pos << endl;
	}

	float dim_y = 0.100 + 0.114;//TODO
	float dim_x = 0.211 + 0.114;
	cv::Vec2f targets[4] = { cv::Vec2f(0,0), cv::Vec2f(dim_x, 0), cv::Vec2f(0, dim_y), cv::Vec2f(dim_x, dim_y) };

	// build matrix of coefficients and independent term for linear eq. system
	cv::Mat A(8, 8, CV_64FC1), b(8, 1, CV_64FC1), x(8, 1, CV_64FC1);

	cv::Vec2f tmp[4];
	for (int i = 0; i < 4; i++) tmp[i] = cv::Vec2f(circle_poses[i].pos(0), circle_poses[i].pos(1)) / circle_poses[i].pos(2);
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
	coordinates_transform = x.reshape(1, 3);
	cout << "H " << coordinates_transform << endl;

	// TODO: compare H obtained by OpenCV with the hand approach
	std::vector<cv::Vec2f> src(4), dsts(4);
	for (int i = 0; i < 4; i++) {
		src[i] = tmp[i];
		dsts[i] = targets[i];
		cout << tmp[i] << " -> " << targets[i] << endl;
	}
	cv::Matx33f H = cv::findHomography(src, dsts, CV_LMEDS);
	cout << "OpenCV H " << H << endl;

	if (!file.empty()) {
		cv::FileStorage fs(file, cv::FileStorage::WRITE);
		fs << "H" << cv::Mat(cv::Matx33d(coordinates_transform)); // store as double to get more decimals
		fs << "c0"; origin_circles[0].write(fs);
		fs << "c1"; origin_circles[1].write(fs);
		fs << "c2"; origin_circles[2].write(fs);
		fs << "c3"; origin_circles[3].write(fs);
	}
	axis_set = true;
	return true;
}

void cv::LocalizationSystem::read_axis(const std::string& file) {
	cv::FileStorage fs(file, cv::FileStorage::READ);
	cv::Mat m;
	fs["H"] >> m;
	coordinates_transform = cv::Matx33f(m);
	origin_circles[0].read(fs["c0"]);
	origin_circles[1].read(fs["c1"]);
	origin_circles[2].read(fs["c2"]);
	origin_circles[3].read(fs["c3"]);
	axis_set = true;
	cout << "transformation: " << coordinates_transform << endl;
}

void cv::LocalizationSystem::draw_axis(cv::Mat& image)
{
	static std::string names[4] = { "0,0", "1,0", "0,1", "1,1" };
	for (int i = 0; i < 4; i++) {
		std::ostringstream ostr;
		//ostr << std::fixed << std::setprecision(5) << names[i] << endl << get_pose(origin_circles[i]).pos;
		origin_circles[i].draw(image, /*ostr.str()*/names[i], cv::Vec3b((i == 0 || i == 3 ? 255 : 0), (i == 1 ? 255 : 0), (i == 2 || i == 3 ? 255 : 0)));
	}

	cv::arrowedLine(image, cv::Point(origin_circles[0].x, origin_circles[0].y), cv::Point(origin_circles[1].x, origin_circles[1].y),
		cv::Scalar(255, 0, 0, 0), 1, 8, 0);
	cv::arrowedLine(image, cv::Point(origin_circles[0].x, origin_circles[0].y), cv::Point(origin_circles[2].x, origin_circles[2].y),
		cv::Scalar(0, 255, 0, 0), 1, 8, 0);
	cv::arrowedLine(image, cv::Point(origin_circles[0].x, origin_circles[0].y), cv::Point(origin_circles[3].x, origin_circles[3].y),
		cv::Scalar(0, 255, 255, 0), 1, 8, 0);
}

/* normalize coordinates: move from image to canonical and remove distortion */
void cv::LocalizationSystem::transform(float x_in, float y_in, float& x_out, float& y_out) const
{
#if defined(ENABLE_FULL_UNDISTORT)
	x = (x - cc[0]) / fc[0];
	y = (y - cc[1]) / fc[1];
#else
	std::vector<cv::Vec2f> src(1, cv::Vec2f(x_in, y_in));
	std::vector<cv::Vec2f> dst(1);
	cv::undistortPoints(src, dst, K, dist_coeff);
	x_out = dst[0](0); y_out = dst[0](1);
#endif
}

void cv::LocalizationSystem::load_opencv_calibration(const std::string& calib_file, cv::Mat& K, cv::Mat& dist_coeff) {
	cv::FileStorage file(calib_file, cv::FileStorage::READ);

	if (!file.isOpened())
	{
		throw std::runtime_error("calibration file not found");
	}//

	std::cout << "file exit!" << std::endl;

	//file["K"] >> K;
	//file["dist"] >> dist_coeff;
	{
		file["Camera_Matrix"] >> K;
		file["Distortion_Coefficients"] >> dist_coeff;
		std::cout << "calibrated result: cameraMatrix : " << K << std::endl;
		std::cout << "distcoeff : " << dist_coeff << std::endl;
		return;
	}

}

void cv::LocalizationSystem::precompute_undistort_map(void)
{
	undistort_map.create(height, width, CV_32FC2);
	for (int i = 0; i < height; i++) {
		std::vector<cv::Vec2f> coords_in(width);
		for (int j = 0; j < width; j++)
			coords_in[j] = cv::Vec2f(j, i); // TODO: reverse y? add 0.5?

		undistortPoints(coords_in, undistort_map.row(i), K, dist_coeff);
	}
}
