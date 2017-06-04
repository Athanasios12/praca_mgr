#include <stdio.h>
#include <CL/cl.h>
#include "CImg.h"
#include <time.h>
#include <memory>
#include <string>
#include <vector>
#include "Viterbi.h"
#include <iostream>
#include <numeric>
#include <algorithm>
#include <stdlib.h>

using namespace cimg_library;
using namespace std;

//image settings
const char IMG_FILE[] = "line_error_test_2.bmp";
const int G_LOW = -8;
const int G_HIGH = 8;

void rgb2Gray(const CImg<unsigned char> &img, CImg<unsigned char> &grayImg)
{
	cimg_forXY(img, x, y)
	{
		int R = (int)img(x, y, 0, 0);
		int G = (int)img(x, y, 0, 1);
		int B = (int)img(x, y, 0, 2);

		// obtain gray value from different weights of RGB channels
		//int gray_value = (int)(0.299 * R + 0.587 * G + 0.114 * B);
		int gray_value = (int)((R + G + B) / 3);
		grayImg(x, y, 0, 0) = gray_value;
	}
}

int initializeCL(cl_command_queue &command_queue, cl_context &context, cl_device_id &device_id)
{
	cl_int err;
	cl_uint numPlatforms;
	cl_platform_id platform_id = NULL;

	err = clGetPlatformIDs(1, &platform_id, &numPlatforms);
	if (CL_SUCCESS == err)
	{
		printf("\nDetected OpenCl platforms :%d", numPlatforms);
	}
	else
	{
		printf("\nError calling get platforms. error code :%d", err);
		getchar();
		return 0;
	}

	//selecting device id - default - gpu
	device_id = NULL;
	cl_uint numOfDevices = 0;

	err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &numOfDevices);
	if (CL_SUCCESS == err)
	{
		printf("\nSelected device id :%d", device_id);
	}
	else
	{
		printf("\nError calling getdeviceIds. error code :%d", err);
		return err;
	}

	//creating context
	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
	if (CL_SUCCESS == err)
	{
		printf("\nCreated context");
	}
	else
	{
		printf("\nError creating context. error code :%d", err);
		return err;
	}

	//creating command queue 
	command_queue = clCreateCommandQueue(context, device_id, 0, &err);
	if (CL_SUCCESS == err)
	{
		printf("\nCreated command queue");
	}
	else
	{
		printf("\nError creating command queue. error code :%d", err);
		return err;
	}
	return CL_SUCCESS;
}

//maybe add clr tabel creation for python/matlab representation or gnuplot
typedef struct
{
	uint32_t m_img_num;
	std::string m_img_name;
	double m_img_size;
	int m_g_low;
	int m_g_high;
	std::vector<double> m_exec_time;
	std::vector<std::vector<unsigned int> > m_lines_pos;
	uint32_t m_detectionError;
}PlotData;

typedef struct
{
	std::vector<PlotData> m_pData;
	uint32_t m_plot_id;
}PlotInfo;

void displayTrackedLine(CImg<unsigned char> &img, const std::vector<unsigned int> &line_x, uint8_t n)
{
	for (int i = 0; i < img.width(); i++)
	{
		img(i, (int)line_x[i], 0, 0) = 255 - (10 * n);
		img(i, (int)line_x[i], 0, 1) = 0 + (10 * n);
		img(i, (int)line_x[i], 0, 2) = 80 + (7 * n);
	}
}

bool readTestLine(const std::string &fileName, std::vector<uint32_t> &test_line)
{
	std::ifstream test_file(fileName);
	std::string line;
	if (test_file.is_open())
	{
		test_line.clear();
		while (std::getline(test_file, line))
		{
			test_line.push_back(static_cast<uint32_t>(std::stoi(line)));
		}
		test_file.close();
	}
	else
	{
		return false;
	}
	
	return true;
}

uint32_t checkDetectionError(const std::vector<unsigned int> &line, const std::vector<uint32_t> &test_line)
{
	uint32_t error = 0;
	if (line.size() == test_line.size())
	{
		for (uint32_t i = 0; i < test_line.size(); i++)
		{
			error += uint32_t(abs(int(test_line[i]) - int(line[i])));
		}
	}
	if (error == 0)
	{
		cout << " ";
	}
	return error;
}

//maybe later add returning error codes
void test_viterbi(const std::vector<std::string> &img_files, int g_h, int g_l, int g_incr, PlotInfo &plotInfo, const std::vector<std::vector<uint32_t> > &test_lines)
{
	if (g_h <= g_l || g_h < 0 || g_l > 0)
	{
		return;
	}
	cl_device_id device_id = NULL;
	cl_command_queue command_queue = NULL;
	cl_context context = NULL;
	//initalize OpenCL 
	if (CL_SUCCESS != initializeCL(command_queue, context, device_id))
	{
		printf("\nFailed OpenCl initialization!\n");
		return;
	}
	Viterbi viterbi(command_queue, context, device_id);
	uint32_t img_num = 0;
	for (auto && img_file : img_files)
	{
		CImg<unsigned char> img(img_file.c_str());
		CImg<unsigned char> gray_img(img.width(), img.height(), 1, 1, 0);
		//convert to monochrome image
		rgb2Gray(img, gray_img);
		std::unique_ptr<unsigned char> img_out(new unsigned char[img.width() * img.height()]);
		memcpy(img_out.get(), gray_img.data(0, 0, 0, 0), img.width() * img.height());

		PlotData data;
		data.m_img_name = img_file.substr(0, img_file.find("."));
		data.m_img_num = img_num;
		data.m_img_size = double(img.width() * img.height()) / double(1024 * 1024);

		viterbi.setImg(gray_img, img.height(), img.width());
		uint8_t n = 0;
		int g_low = g_l;
		int g_high = g_h;
		while (g_high + abs(g_low) > 0 && g_low <= 0 && g_high >= 0) // maybe redundant but whatever...
		{
			std::vector<std::vector<unsigned int> > line_results;
			std::vector<double> times;
			std::vector<unsigned int> line_x(img.width(), 0);
			std::vector<uint32_t> detectionError;

			//viterbi parallel cols gpu version 
			clock_t start = clock();
			viterbi.viterbiLineOpenCL_cols(&line_x[0], g_low, g_high);
			clock_t end = clock();
			double time_ms = (double)(end - start);
			detectionError.push_back(checkDetectionError(line_x, test_lines[img_num]));
			line_results.push_back(line_x);
			times.push_back(time_ms);
			cout << "\nGPU time :" << time_ms << endl;
			//viterbi serial cpu version
			start = clock();
			viterbi.viterbiLineDetect(line_x, g_low, g_high);
			end = clock();
			time_ms = (double)(end - start);
			//check line detection error based on desired cordinates from line_test.py script, save it in another column
			detectionError.push_back(checkDetectionError(line_x, test_lines[img_num]));
			line_results.push_back(line_x);
			times.push_back(time_ms);
			cout << "\nSerial time :" << time_ms << endl;
			/*
			gpu rows version
			start = clock();
			//viterbi parallel rows gpu version
			viterbi.viterbiLineOpenCL_rows(&line_x[0], g_low, g_high);
			end = clock();
			time_ms = (double)(end - start);

			line_results.push_back(line_x);
			times.push_back(time_ms);
			*/

			//cpu multithread version
			start = clock();
			viterbi.launchViterbiMultiThread(line_x, g_low, g_high);
			end = clock();
			time_ms = (double)(end - start);
			detectionError.push_back(checkDetectionError(line_x, test_lines[img_num]));
			line_results.push_back(line_x);
			times.push_back(time_ms);
			cout << "\nThreads time :" << time_ms << endl;
			//save data for ploting and tabel representation
			data.m_g_high = g_high;
			data.m_g_low = g_low;
			data.m_exec_time = times;
			data.m_lines_pos = line_results;
			uint32_t avg_err = std::accumulate(detectionError.begin(), detectionError.end(), 0) / detectionError.size();
			for (auto && error : detectionError)
			{
				cout << "\nError : " << error;
			}
			data.m_detectionError = avg_err;
			//save plot data in plot info data vector
			plotInfo.m_pData.push_back(data);
			//next g_l/h combo
			displayTrackedLine(img, line_x, n);
			g_low += g_incr;
			g_high -= g_incr;
			++n;
		}
		++img_num;
		img.save_bmp((data.m_img_name + "result.bmp").c_str());
	}
	//cleanup
	int err = 0;
	err = clFlush(command_queue); 
	err = clFinish(command_queue);
	err = clReleaseCommandQueue(command_queue);
	err = clReleaseContext(context);
}

void generateCsv(const std::string &file, const PlotInfo &pInfo, const std::vector<std::string> &columnNames)
{
	std::ofstream data_file;
	data_file.open(file);
	uint32_t i = 0;
	for (auto && column : columnNames)
	{
		if (i > 0)
		{
			data_file << "," << column;
		}
		else
		{
			data_file << column;
		}
		++i;
	}
	data_file << "\n";
	for (auto && data : pInfo.m_pData)
	{
		data_file << data.m_img_num << "," << data.m_img_size << "," << data.m_g_low << "," << data.m_g_high << "," << data.m_detectionError;
		for (auto && time : data.m_exec_time)
		{
			data_file << "," << time;
		}
		data_file << "\n";
	}
}

void basicTest()
{
	cl_device_id device_id = NULL;
	cl_command_queue command_queue = NULL;
	cl_context context = NULL;
	//initalize OpenCL 
	if (CL_SUCCESS != initializeCL(command_queue, context, device_id))
	{
		printf("\nFailed OpenCl initialization!\n");
return;
	}
	CImg<unsigned char> img(IMG_FILE);
	CImg<unsigned char> gray_img(img.width(), img.height(), 1, 1, 0);
	//convert to monochrome image
	rgb2Gray(img, gray_img);

	std::unique_ptr<unsigned char> img_out(new unsigned char[img.width() * img.height()]);
	memcpy(img_out.get(), gray_img.data(0, 0, 0, 0), img.width() * img.height());

	std::vector<unsigned int> line_x(img.width());
	Viterbi viterbi(command_queue, context, device_id);
	viterbi.setImg(gray_img, img.height(), img.width());
	clock_t start = clock();
	viterbi.viterbiLineOpenCL_cols(&line_x[0], G_LOW, G_HIGH);
	clock_t end = clock();
	double time_ms = (double)(end - start);
	for (int i = 0; i < img.width(); i++)
	{
		img(i, (int)line_x[i], 0, 0) = 255;
		img(i, (int)line_x[i], 0, 1) = 0;
		img(i, (int)line_x[i], 0, 2) = 0;
	}
	CImgDisplay gray_disp(gray_img, "image gray");
	CImgDisplay rgb_disp(img, "image rgb1");
	printf("\nviterbi parallel time, cols version: %f ms\n", time_ms);

	//reset line_x 
	for (uint32_t i = 0; i < img.width(); i++)
	{
		line_x[i] = 0;
	}

	start = clock();
	viterbi.viterbiLineDetect(line_x, G_LOW, G_HIGH);
	end = clock();
	time_ms = (double)(end - start);
	for (int i = 0; i < img.width(); i++)
	{
		img(i, (int)line_x[i], 0, 0) = 0;
		img(i, (int)line_x[i], 0, 1) = 255;
		img(i, (int)line_x[i], 0, 2) = 0;
	}
	printf("\nViterbi serial time: %f ms\n", time_ms);
	CImgDisplay rgb1_disp(img, "Image rgb2");

	//reset line_x 
	for (uint32_t i = 0; i < img.width(); i++)
	{
		line_x[i] = 0;
	}

	start = clock();
	viterbi.launchViterbiMultiThread(line_x, G_LOW, G_HIGH);
	end = clock();
	time_ms = (double)(end - start);
	for (int i = 0; i < img.width(); i++)
	{
		img(i, (int)line_x[i], 0, 0) = 0;
		img(i, (int)line_x[i], 0, 1) = 0;
		img(i, (int)line_x[i], 0, 2) = 255;
	}
	printf("\nViterbi parallel time , threads CPU version: %f ms\n", time_ms);
	CImgDisplay rgb2_disp(img, "Image rgb3");
	img.save_bmp("threds1.bmp");

	/*start = clock();
	viterbiLineOpenCL_rows(img_out.get(), img.height(), img.width(), &line_x[0], G_LOW, G_HIGH, command_queue, context, device_id);
	end = clock();
	time_ms = (double)(end - start);
	for (int i = 0; i < img.width(); i++)
	{
	img(i, (int)line_x[i], 0, 0) = 0;
	img(i, (int)line_x[i], 0, 1) = 0;
	img(i, (int)line_x[i], 0, 2) = 255;
	}
	printf("\nViterbi parallel time , rows version: %f ms\n", time_ms);
	CImgDisplay rgb2_disp(img, "Image rgb3");*/
	while (!rgb2_disp.is_closed());
	//cleanup
	int err = 0;
	err = clFlush(command_queue);
	err = clFinish(command_queue);
	err = clReleaseCommandQueue(command_queue);
	err = clReleaseContext(context);
}

int main(void)
{
	//basic tests, later call test viterbi
	PlotInfo pInfo;
	//declare list of images - maybe load from file later
	std::vector<std::string> images{ "line_error_test_0.bmp", "line_error_test_1.bmp", "line_error_test_2.bmp" };// "line_error_test_0.bmp", "line_error_test_1.bmp" , 

	//basicTest();
	std::vector<std::vector<uint32_t> > test_lines;
	std::vector<uint32_t> test_line;
	char buff[100];
	bool success = true;
	for (uint32_t i = 0; i < images.size(); i++)
	{
		_itoa_s(i, buff, 10);
		std::string file_name = "line_pos_" + std::string(buff) + ".txt";
		success = readTestLine(file_name, test_line);
		test_lines.push_back(test_line);
	}
	if (success)
	{
		test_viterbi(images, G_HIGH, G_LOW, 1, pInfo, test_lines); // init g_low, g_high = <-5, 5>
		//check inside the test viterbi function if errors are indeed the same, if not something is not right, because algorithms results should be indentical
		//error should be the same for all algorithms, only time should be different
		std::vector<std::string> columns{ "img_num", "img_size", "g_low", "g_high", "error", "GPU", "SERIAL", "CPU_THREADS" };											
		generateCsv("../../test_results_clang.csv", pInfo, columns);
	}
	else
	{
		cout << "Failed reading test line file" << endl;
	}

	return 0;
}