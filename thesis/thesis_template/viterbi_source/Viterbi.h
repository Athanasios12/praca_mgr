#ifndef VITERBI_H
#define VITERBI_H

#include <stdio.h>
#include <CL/cl.h>
#include "CImg.h"
#include <vector>
#include <memory>
#include <fstream>
#include <string>
#include <future>

using namespace cimg_library;

//kernel constants
const char VITERBI_KERNEL_FILE[] = "viterbi_kernel.cl"; 
const char VITERBI_COLS_FUNCTION[] = "viterbi_function"; 
const char VITERBI_GPU_FRAGMENT_FUNCTION[] = "viterbi_gpu_fragment";

class Viterbi
{
public:
	Viterbi(const cl_command_queue &command_queue,
		const cl_context &context,
		cl_device_id device_id);
	~Viterbi();

	//public methods
	void setImg(const unsigned char *img, size_t img_height, size_t img_width);
	int viterbiLineDetect(std::vector<unsigned int> &line_x, int g_low, int g_high);
	int viterbiLineOpenCL_cols(unsigned int *line_x, int g_low, int g_high);
	int launchViterbiMultiThread(std::vector<unsigned int>& line_x, int g_low, int g_high);
	bool launchHybridViterbi(std::vector<unsigned int>& line_x, int g_low, int g_high);
	bool viterbiOpenMP(std::vector<unsigned int> &line_x, int g_low, int g_high);
	bool launchHybridViterbiOpenMP(std::vector<unsigned int> &line_x, int g_low, int g_high);
private:
	//private methods
	size_t readKernelFile(std::string &source_str, const std::string &fileName);
	void fixGlobalSize(size_t &global_size, const size_t &local_size);
	unsigned int viterbiMultiThread(int g_low, int g_high, unsigned int start_col);
	bool loadAndBuildKernel();
	double viterbiHybridCPU(std::vector<unsigned int> &line_x, int g_low, int g_high, uint32_t start_col, uint32_t end_col);
	double viterbiHybridGPU(unsigned int *line_x, int g_low, int g_high, uint32_t start_col, uint32_t end_col);
	double viterbiHybridOpenMP_CPU(std::vector<unsigned int> &line_x, int g_low, int g_high, uint32_t start_col, uint32_t end_col);

	//private memebers
	const unsigned char *m_img;
	size_t m_img_width;
	size_t m_img_height;
	bool m_size_changed;
	cl_command_queue m_command_queue;
	cl_context m_context;
	cl_device_id m_device_id;
	bool m_initalized = false;
	cl_program m_program;
	cl_kernel m_viterbiKernel;
	cl_kernel m_viterbiHybridKernel;
	bool m_set_hybrid_rate;
	std::pair<double, double> m_hybrid_rate;
};

#endif //VITERBI_H