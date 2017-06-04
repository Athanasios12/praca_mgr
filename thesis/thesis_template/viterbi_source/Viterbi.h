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

const size_t MAX_SOURCE_SIZE = 0x100000;
const char VITERBI_KERNEL_FILE[] = "viterbi_kernel.cl"; // later pass file name as argument in contructor
const char VITERBI_COLS_FUNCTION[] = "viterbi_function"; //pass kernel function name in 
const char VITERBI_ROWS_FUNCTION[] = "viterbi_forward";

class Viterbi
{
public:
	
	Viterbi(const cl_command_queue &command_queue, 
			const cl_context &context, 
			cl_device_id device_id);

	~Viterbi();
	
	int viterbiLineDetect(std::vector<unsigned int> &line_x, int g_low, int g_high);
	int viterbiLineOpenCL_cols(unsigned int *line_x, int g_low, int g_high);
	void setImg(const unsigned char *img, size_t img_height, size_t img_width);

	int launchViterbiMultiThread(std::vector<unsigned int>& line_x, int g_low, int g_high);

private:
	//methods
	//fix for later
	int viterbiLineOpenCL_rows(unsigned int *line_x, int g_low, int g_high);
	//works
	size_t readKernelFile(std::string &source_str, const std::string &fileName);
	void fixGlobalSize(size_t &global_size, const size_t &local_size);
	unsigned int viterbiMultiThread(int g_low, int g_high, unsigned int start_col);
	bool loadAndBuildKernel();
	
	//class memebers
	const unsigned char *m_img;
	size_t m_img_width;
	size_t m_img_height;
	cl_command_queue m_command_queue;
	cl_context m_context;
	cl_device_id m_device_id;
	bool m_initalized = false;
	cl_program m_program;
	cl_kernel m_viterbiKernel;
};

#endif //VITERBI_H