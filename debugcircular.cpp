#include "opencv2/opencv.hpp"  
#include <string>  
#include <vector>  
#include <memory>  
#include <algorithm>  
#include <opencv2/opencv.hpp>  
#include <opencv2/ml.hpp>  
#include <chrono>
//#include "opencv2/core/cuda/common.hpp"  
#include <set>
#include <direct.h>
#include <io.h>
#include <iostream>
#include <algorithm>
#include <windows.h>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <set>
#include <fstream>
#include <iomanip>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <omp.h>

#include "circular.h"
#include "refinedCircularPositioningSystem.h"
#ifdef _DEBUG
#pragma comment(lib, "opencv_core310d.lib")
#pragma comment(lib, "opencv_videoio310d.lib")
#pragma comment(lib, "opencv_highgui310d.lib")
#pragma comment(lib, "opencv_calib3d310d.lib")
#pragma comment(lib, "opencv_imgproc310d.lib")
#pragma comment(lib, "opencv_imgcodecs310d.lib")
#pragma comment(lib, "opencv_tracking310d.lib")
#pragma comment(lib, "opencv_video310d.lib")
#pragma comment(lib, "opencv_ml310d.lib")
#else
#pragma comment(lib, "opencv_core310.lib")
#pragma comment(lib, "opencv_videoio310.lib")
#pragma comment(lib, "opencv_highgui310.lib")
#pragma comment(lib, "opencv_calib3d310.lib")
#pragma comment(lib, "opencv_imgproc310.lib")
#pragma comment(lib, "opencv_imgcodecs310.lib")
#pragma comment(lib, "opencv_tracking310.lib")
#pragma comment(lib, "opencv_video310.lib")
#pragma comment(lib, "opencv_ml310.lib")
#endif

using namespace std;

#pragma warning(disable:4305)

string getFileSuffix(string filename)
{
	string suffix;
	suffix = filename.substr(filename.find_last_of('.') + 1);//获取文件后缀  
	return suffix;
}

/*
* @function: 获取cate_dir目录下的所有文件名
* @param: cate_dir - string类型
* @result：vector<string>类型
*/
vector<string> getFiles(string cate_dir)
{
	vector<string> files;//存放文件名  

#ifdef WIN32  
	_finddata_t file;
	long lf;
	//输入文件夹路径  
	if ((lf = _findfirst(cate_dir.c_str(), &file)) == -1) {
		cout << cate_dir << " not found!!!" << endl;
	}
	else {
		while (_findnext(lf, &file) == 0) {
			//输出文件名  
			//cout<<file.name<<endl;  
			if (strcmp(file.name, ".") == 0 || strcmp(file.name, "..") == 0)
				continue;
			files.push_back(file.name);
		}
	}
	_findclose(lf);
#endif  

#ifdef linux  
	DIR *dir;
	struct dirent *ptr;
	char base[1000];

	if ((dir = opendir(cate_dir.c_str())) == NULL)
	{
		perror("Open dir error...");
		exit(1);
	}

	while ((ptr = readdir(dir)) != NULL)
	{
		if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0)    ///current dir OR parrent dir  
			continue;
		else if (ptr->d_type == 8)    ///file  
									  //printf("d_name:%s/%s\n",basePath,ptr->d_name);  
			files.push_back(ptr->d_name);
		else if (ptr->d_type == 10)    ///link file  
									   //printf("d_name:%s/%s\n",basePath,ptr->d_name);  
			continue;
		else if (ptr->d_type == 4)    ///dir  
		{
			files.push_back(ptr->d_name);
			/*
			memset(base,'\0',sizeof(base));
			strcpy(base,basePath);
			strcat(base,"/");
			strcat(base,ptr->d_nSame);
			readFileList(base);
			*/
		}
	}
	closedir(dir);
#endif  

	//排序，按从小到大排序  
	sort(files.begin(), files.end());
	return files;
}

#define IMG_DIR		"C:\\Users\\xiahaa\\workspace\\data\\offline\\174"
#define CALI_FILE	"C:/Users/xiahaa/workspace/mvBlueFox/cali_param_2_22_12_24.yml"
#define AXIS_FILE	"C:/Users/xiahaa/workspace/circular/axis.yml"

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

bool readCaliFile(cv::Mat &camMatrix, cv::Mat &camDistCoeff)
{
	if (1)
	{
		cv::FileStorage fs(CALI_FILE, cv::FileStorage::READ);
		if (!fs.isOpened())
		{
			std::printf("cali file missing......!!!!!\n");
			return false;
		}
		else
		{
			fs["Camera_Matrix"] >> camMatrix;
			fs["Distortion_Coefficients"] >> camDistCoeff;
			std::cout << "calibrated result: cameraMatrix : " << camMatrix << std::endl;
			std::cout << "distcoeff : " << camDistCoeff << std::endl;
			return true;
		}
	}
	else
	{
		return true;
	}
}


int clicked = 0;
std::vector<cv::Point> pt(4);
void mouse_callback(int event, int x, int y, int flags, void* param) {
	if (event == CV_EVENT_LBUTTONDOWN) 
	{ 
		cout << "clicked window: "<< clicked << " x:"<<x << " y:" << y << endl;
		pt[clicked] = cv::Point(x, y);
		clicked = (clicked+1)>3?0:(clicked+1);
	}
}

int main()
{
	//return 0;
	cv::Size imgSize;
	cout << "searching " << IMG_DIR << endl;
	string rootDir(IMG_DIR);
	string searchDir;
	if (rootDir.back() != '\\' || rootDir.back() != '/')
	{
		rootDir = rootDir + "\\";
		searchDir = rootDir + "*";
	}
	else
	{
		searchDir = rootDir + "*";
	}
	vector<string> files;
	files = getFiles(searchDir);
	vector<string> imgfiles;
	imgfiles.clear();
	for (vector<string>::iterator it = files.begin(); it != files.end(); it++)
	{
		string suffix = getFileSuffix(*it);
		if (suffix == "png" || suffix == "bmp")
		{
			cout << rootDir + *it << endl;
			imgfiles.push_back(rootDir + *it);
		}
	}
	cout << "There are " << imgfiles.size() << " images." << endl;

	cv::namedWindow("raw image", cv::WINDOW_NORMAL);
	cv::resizeWindow("raw image", 640, 480);

	/* setup gui */
	if (1) {
		cvStartWindowThread();
		cv::namedWindow("frame", CV_WINDOW_NORMAL);
		cv::setMouseCallback("frame", mouse_callback);
	}

	/* load calibration */
	cv::Mat K, dist_coeff;
	readCaliFile(K, dist_coeff);

	bool init = false;
	/* init system */
	cv::Size frame_size;
	cv::LocalizationSystem system;

	float inner_diameter = 0.047;//0.047;/0.023
	float outer_diameter = 0.114;////0.056
	float dim_x = 0.049 + 0.056;//TODO
	float dim_y = 0.062 + 0.056;
	bool axisSet = false;
	string axisfile(AXIS_FILE);
	int num_of_markers = 1;
	bool do_tracking = false;
	circularPatternBasedLocSystems cplocsys(K, dist_coeff, inner_diameter, outer_diameter, dim_x, dim_y);

	cv::Mat frame, frame_gray;
	bool is_tracking = false;

#define ENABLE_LOG_DUMP	1
#if ENABLE_LOG_DUMP
	std::ofstream logfile;
	std::string storageLog = rootDir + "position.log";
	logfile.open(storageLog, std::ios::out);
	if (logfile.is_open())
	{
		std::cout << "INFO: positioning Logfile ready." << std::endl;
		logfile << "positioning LOG START" << std::endl;
	}
	else
	{
		std::cout << "ERROR: Unable to open logfile. Logging disabled." << endl;
		return false;
	}
#endif

	int frame_idx = 0;
	for (vector<string>::iterator it = imgfiles.begin(); it != imgfiles.end(); it++)
	{
		//frame.setTo(0);
		frame_gray = cv::imread(*it, 0);
		cv::cvtColor(frame_gray, frame, cv::COLOR_GRAY2BGR);
#if 1
		/* reform as an independent function */
		std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
		int detected_markers = cplocsys.detectPatterns(frame_gray, do_tracking);
		if (detected_markers == num_of_markers)do_tracking = true;
		else do_tracking = false;
		std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
		cplocsys.localization();
		std::chrono::high_resolution_clock::time_point t3 = std::chrono::high_resolution_clock::now();
		std::cout << "t2-t1:"<< std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << std::endl;
		std::cout <<"t3-t2:"<< std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count() << std::endl;
		cplocsys.drawPatterns(frame);

		if (axisSet == false && 0)
		{
			cv::imshow("frame", frame);
			char key = cv::waitKey(1);
			if (key == 's')
			{
				cplocsys.setAxisFrame(pt,axisfile);
				cplocsys.draw_axis(frame);
				cv::imshow("frame", frame);
				char key = cv::waitKey(0);
				if (key == 'o')
				{
					axisSet = true;
				}
			}
		}
		else
		{
			cv::imshow("frame", frame);
			char key = cv::waitKey(0);
			logfile << frame_idx++ << ":";
			cplocsys.tostream(logfile);
			//cv::namedWindow("tracking result", CV_WINDOW_NORMAL);
			//cv::moveWindow("tracking result", 100, 100);
			//cv::imshow("tracking result", frame);
			//cv::waitKey(0);
		}

		//cv::destroyAllWindows();
#endif

#if 0
		if (init == false)
		{
			init = true;
			frame_size = cv::Size(frame_gray.cols, frame_gray.rows);
			std::cout << "frame size: " << frame_size << std::endl;
			system.setParams(1, frame_size.width, frame_size.height, K, dist_coeff, outer_diameter, inner_diameter);
			system.setDimXY(dim_x, dim_y);
		}

		if (axisSet == true)
		{
			// TODO, read first, if not exist, then do axis setting

			// firstly, set axis, or just use camera coordinate
			axisSet = system.set_axis(frame, 5, 15, AXIS_FILE);
			system.draw_axis(frame);
		}
		else
		{
			int number_of_targets = 1;
			if (!is_tracking) cout << "resetting targets" << endl;

			int64_t ticks = cv::getTickCount();
			is_tracking = system.localize(frame, !is_tracking, 5, 15);
			double delta_ticks = (double)(cv::getTickCount() - ticks) / cv::getTickFrequency();

			cout << "localized all? " << is_tracking << " t: " << delta_ticks << " " << " fps: " << 1 / delta_ticks << endl;



			if (is_tracking) {
				for (int i = 0; i < number_of_targets; i++) {
					const cv::CircleDetector::Circle& circle = system.get_circle(i);
					cv::Vec3f coord = system.get_pose(circle).pos;
					cv::Vec3f coord_trans = coord;
					if (axisSet) {
						//coord_trans = system.get_transformed_pose(circle).pos;
					}

					if (1) {
						ostringstream ostr;
						ostr << fixed << setprecision(2);
						ostr << coord_trans[0] << ";" << coord_trans[1] << ";" << coord_trans[2];
						//ostr << coord_trans[1] << endl;
						circle.draw(frame, ostr.str(), cv::Vec3b(0, 0, 255));
						//stringstream ss;
						//ss << "x:" << setw(6) << coord_trans[0] << "y:" << setw(6) << coord_trans[1] << std::endl;
						//cv::Point2i pt;
						//pt = circle.returnEllipseCenter();
						//cv::putText(frame, ss.str(), cv::Point(pt.x+10, pt.y+10), CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255,255,255,255), 1, CV_AA);

					}
				}
			}
		}
#endif

		cv::imshow("frame", frame);

		//frame.release();
		if ((cv::waitKey(1) & 255) == 'q')
		{
			break;
		}
	}

#if ENABLE_LOG_DUMP
	logfile.close();
#endif

	return 0;
}