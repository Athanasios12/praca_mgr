#include "Viterbi.h"
#include <iostream>
#include <mutex>

using namespace std;

Viterbi::Viterbi(const cl_command_queue & command_queue, const cl_context & context, cl_device_id device_id):
	m_img(NULL),
	m_command_queue(command_queue),
	m_context(context),
	m_device_id(device_id),
	m_set_hybrid_rate(false),
	m_hybrid_rate(std::pair<double, double>(0.5, 0.5))
{ 
	m_initalized = loadAndBuildKernel();
}

Viterbi::~Viterbi()
{
	if (m_initalized)
	{
		clReleaseKernel(m_viterbiKernel);
		clReleaseProgram(m_program);
	}
}

bool Viterbi::loadAndBuildKernel()
{
	int err = 0;
	std::string source_str;
	// Load the source code containing the kernel
	size_t source_size = readKernelFile(source_str, VITERBI_KERNEL_FILE);
	if (source_size == 0)
	{
		return false;
	}
	char *source_str_ptr = &source_str[0];
	 m_program = clCreateProgramWithSource(m_context, 1, (const char **)&source_str_ptr,
		(const size_t *)&source_size, &err);
	if (CL_SUCCESS != err)
	{
		return false;
	}
	// Build Kernel Program
	err = clBuildProgram(m_program, 1, &m_device_id, NULL, NULL, NULL);
	if (err == CL_BUILD_PROGRAM_FAILURE) {
		// Determine the size of the log
		size_t log_size;
		clGetProgramBuildInfo(m_program, m_device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

		// Allocate memory for the log
		char *log = (char *)malloc(log_size);

		// Get the log
		clGetProgramBuildInfo(m_program, m_device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

		// Print the log
		printf("%s\n", log);
		free(log);
	}
	// Create OpenCL Kernel
	m_viterbiKernel = clCreateKernel(m_program, VITERBI_COLS_FUNCTION, &err);
	return err == CL_SUCCESS;
}

size_t Viterbi::readKernelFile(std::string &source_str, const std::string &fileName)
{
	std::ifstream file(fileName, std::ifstream::binary);
	std::string line;
	source_str.clear();
	size_t length = 0;
	if (file.is_open())
	{
		file.seekg(0, file.end);
		length = file.tellg();
		file.seekg(0, file.beg);
		std::unique_ptr<char> buffer(new char[length]);
		// read data as a block:
		file.read(buffer.get(), length);
		source_str = std::string(buffer.get());
		file.close();
	}
	return length;
}

void Viterbi::fixGlobalSize(size_t &global_size, const size_t &local_size)
{
	if (global_size % local_size != 0)
	{
		size_t multiple = global_size / local_size;
		++multiple;
		global_size = multiple * local_size;
	}
}

//obsolete , needs fixing, maybe later
int Viterbi::viterbiLineOpenCL_rows(unsigned int *line_x, int g_low, int g_high)
{
	//read kernel file
	int err = 0;
	std::string source_str;
	size_t img_size = (m_img_height * m_img_width);
	// Load the source code containing the kernel*/
	size_t source_size = readKernelFile(source_str, VITERBI_KERNEL_FILE);
	if (source_size == 0)
	{
		return 1;
	}
	cl_program program = clCreateProgramWithSource(m_context, 1, (const char **)&source_str[0],
		(const size_t *)&source_size, &err);
	if (CL_SUCCESS != err)
	{
		return 0;
	}
	// Build Kernel Program */
	err = clBuildProgram(program, 1, &m_device_id, NULL, NULL, NULL);
	if (err == CL_BUILD_PROGRAM_FAILURE) {
		// Determine the size of the log
		size_t log_size;
		clGetProgramBuildInfo(program, m_device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

		// Allocate memory for the log
		char *log = (char *)malloc(log_size);

		// Get the log
		clGetProgramBuildInfo(program, m_device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

		// Print the log
		printf("%s\n", log);
		free(log);
	}

	// Create OpenCL Kernel */
	cl_kernel viterbi_forward = clCreateKernel(program, VITERBI_ROWS_FUNCTION, &err);

	size_t global_size = m_img_height;

	std::vector<float> L(img_size, 0);
	std::vector<float> V_old(m_img_height, 0);
	std::vector<float> V_new(m_img_height, 0);

	cl_mem cmImg = clCreateBuffer(m_context, CL_MEM_READ_ONLY, sizeof(unsigned char) * img_size, NULL, &err);
	err = clEnqueueWriteBuffer(m_command_queue, cmImg, CL_FALSE, 0, sizeof(unsigned char) * img_size, m_img, 0, NULL, NULL);

	cl_mem cmV_old = clCreateBuffer(m_context, CL_MEM_READ_WRITE, m_img_height * sizeof(float), NULL, &err);
	cl_mem cmV_new = clCreateBuffer(m_context, CL_MEM_READ_WRITE, m_img_height * sizeof(float), NULL, &err);
	cl_mem cmL = clCreateBuffer(m_context, CL_MEM_READ_WRITE, img_size * sizeof(float), NULL, &err);


	err = clEnqueueWriteBuffer(m_command_queue, cmV_old, CL_FALSE, 0, sizeof(float) * m_img_height, &V_old[0], 0, NULL, NULL);
	err = clEnqueueWriteBuffer(m_command_queue, cmV_new, CL_FALSE, 0, sizeof(float) * m_img_height, &V_new[0], 0, NULL, NULL);
	err = clEnqueueWriteBuffer(m_command_queue, cmL, CL_FALSE, 0, sizeof(float) * img_size, &L[0], 0, NULL, NULL);

	int start_column = 0;
	err = clSetKernelArg(viterbi_forward, 0, sizeof(cl_mem), (void*)&cmImg);
	err = clSetKernelArg(viterbi_forward, 1, sizeof(cl_mem), (void*)&cmL);
	err = clSetKernelArg(viterbi_forward, 2, sizeof(cl_mem), (void*)&cmV_old);
	err = clSetKernelArg(viterbi_forward, 3, sizeof(cl_mem), (void*)&cmV_new);
	err = clSetKernelArg(viterbi_forward, 4, sizeof(cl_int), (void*)&m_img_height);
	err = clSetKernelArg(viterbi_forward, 5, sizeof(cl_int), (void*)&m_img_width);
	err = clSetKernelArg(viterbi_forward, 6, sizeof(cl_int), (void*)&start_column);
	err = clSetKernelArg(viterbi_forward, 7, sizeof(cl_int), (void*)&g_low);
	err = clSetKernelArg(viterbi_forward, 8, sizeof(cl_int), (void*)&g_high);

	int i = 0;
	float P_max = 0;
	unsigned int x_max = 0;
	std::vector<float> init_V(m_img_height, 0);
	//allocate buffer x_cord
	std::vector<unsigned int> x_cord(m_img_width, 0);

	while (start_column < m_img_width && !err)
	{
		// Execute OpenCL Kernel */
		err |= clEnqueueNDRangeKernel(m_command_queue, viterbi_forward, 1, NULL, &global_size, NULL, 0, NULL, NULL);
		// Copy results from the memory buffer */
		err |= clEnqueueReadBuffer(m_command_queue, cmL, CL_TRUE, 0,
			img_size * sizeof(float), &L[0], 0, NULL, NULL);
		err |= clEnqueueReadBuffer(m_command_queue, cmV_new, CL_TRUE, 0,
			m_img_height * sizeof(float), &V_new[0], 0, NULL, NULL);

		for (int j = 0; j < m_img_height; j++)
		{
			if (V_new[j] > P_max)
			{
				P_max = V_new[j];
				x_max = j;
			}
		}
		//backwards phase - retrace the path
		x_cord[(m_img_width - 1)] = x_max;
		for (size_t n = (m_img_width - 1); n > start_column; n--)
		{
			x_cord[n - 1] = x_cord[n] + static_cast<unsigned int>(L[(x_cord[n] * m_img_width) + (n - 1)]);
		}
		// save only last pixel position
		line_x[start_column] = x_cord[start_column];
		P_max = 0;
		x_max = 0;
		V_old = init_V; // copy elements and init vold with zeros - check if works without it
		++start_column;
		err |= clSetKernelArg(viterbi_forward, 5, sizeof(cl_int), (void*)&start_column); // check if works without
	}
	if (!err)
	{
		line_x[m_img_width - 1] = line_x[m_img_width - 2];
	}
	//realase resources
	err = clReleaseKernel(viterbi_forward);
	err = clReleaseProgram(program);
	err = clReleaseMemObject(cmV_old);
	err = clReleaseMemObject(cmV_new);
	err = clReleaseMemObject(cmL);
	err = clReleaseMemObject(cmImg);
	return err;
}

int Viterbi::viterbiLineDetect(std::vector<unsigned int> &line_x, int g_low, int g_high)
{
	if (m_img == 0 && m_img_height > 0 && m_img_width > 0)
	{
		return 1;
	}
	//allocate array for viterbi algorithm
	std::vector<uint32_t> L(m_img_height * m_img_width, 0);
	std::vector<uint32_t> V(m_img_height * m_img_width, 0);

	uint32_t P_max = 0;
	uint32_t x_max = 0;
	uint32_t max_val = 0;
	size_t i = 0;
	unsigned char pixel_value = 0;
	while (i < (m_img_width - 1))
	{
		// init first column with zeros
		for (size_t m = 0; m < m_img_height; m++)
		{
			V[(m * m_img_width) + i] = 0;
		}
		for (size_t n = i; n < (m_img_width - 1); n++)
		{
			for (int j = 0; j < m_img_height; j++)
			{
				max_val = 0;
				for (int g = g_low; g <= g_high; g++)
				{
					if ((j + g) >(int)(m_img_height - 1))
					{
						break;
					}
					if (j + g < 0)
					{
						continue;
					}
					int curr_id = j + g;
					pixel_value = m_img[((curr_id)* m_img_width) + n];
					if ((pixel_value + V[(m_img_width * curr_id) + n]) > max_val)
					{
						max_val = pixel_value + V[(m_img_width * curr_id) + n];
						L[(j * m_img_width) + n] = g;
					}
				}
				V[(j * m_img_width) + (n + 1)] = max_val;
			}
		}
		//find biggest cost value in last column
		for (size_t j = 0; j < m_img_height; j++)
		{
			if (V[(j * m_img_width) + (m_img_width - 1)] > P_max)
			{
				P_max = V[(j * m_img_width) + (m_img_width - 1)];
				x_max = j;
			}
		}
		//backwards phase - retrace the path
		uint32_t x_n = x_max;
		for (size_t n = (m_img_width - 1); n > i; n--)
		{
			x_n = x_n + L[(x_n * m_img_width) + (n - 1)];
		}
		// save only last pixel position
		line_x[i] = x_n;
		P_max = 0;
		x_max = 0;
		++i;
	}
	line_x[m_img_width - 1] = line_x[m_img_width - 2];
	return 0;
}

int Viterbi::viterbiLineOpenCL_cols(unsigned int *line_x, int g_low, int g_high)
{
	int err = 0;
	if (!m_initalized || m_img == NULL)
	{
		cout << "Kernel not initialized" << endl;
		return err;
	}
	size_t img_size = (m_img_height * m_img_width);
	size_t global_size = m_img_width; 

	//check available memory
	cl_ulong dev_memory = 0;
	err = clGetDeviceInfo(m_device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &dev_memory, NULL);
	cl_ulong max_alloc = 0;
	err = clGetDeviceInfo(m_device_id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &max_alloc, NULL);
	int dev_mem = static_cast<int>(double(dev_memory) / double(1024 * 1024));//MB
	int max_buff_size = static_cast<int>(double(max_alloc) / double(1024 * 1024));
	int tot_mem = static_cast<int>(double((img_size * global_size * sizeof(float)) +
		(2 * m_img_height * global_size * sizeof(float)) +
		(2 * m_img_width * sizeof(int)) + (img_size * sizeof(unsigned char))) / double(1024 * 1024));
#ifdef _DEBUG
	printf("\nMax buffer size: %d MB\n", max_buff_size);
	printf("Total available memory : %d MB\n", dev_mem);
	printf("\nTotal memory used %d MB\n", tot_mem);
	printf("L indices matrixes size : %d MB", int(double(img_size * global_size * sizeof(float) / double(1024 * 1024))));
#endif
	//handle not enough GPU memory
	if (max_buff_size < tot_mem)
	{
		int mem_multiple = (int)(tot_mem / max_buff_size);
		global_size = m_img_width / (mem_multiple + 1);
	}

	//create necessery opencl buffers
	cl_mem cmImg = clCreateBuffer(m_context, CL_MEM_READ_ONLY, sizeof(unsigned char) * img_size, NULL, &err);
	err = clEnqueueWriteBuffer(m_command_queue, cmImg, CL_FALSE, 0, sizeof(unsigned char) * img_size, m_img, 0, NULL, NULL);

	cl_mem cmLine_x = clCreateBuffer(m_context, CL_MEM_READ_WRITE, m_img_width * sizeof(int), NULL, &err);
	cl_mem cmV1 = clCreateBuffer(m_context, CL_MEM_READ_WRITE, m_img_height * global_size * sizeof(float), NULL, &err);
	cl_mem cmV2 = clCreateBuffer(m_context, CL_MEM_READ_WRITE, m_img_height * global_size * sizeof(float), NULL, &err);
	cl_mem cmL = clCreateBuffer(m_context, CL_MEM_READ_WRITE, img_size * global_size * sizeof(float), NULL, &err);

	//set kernel arguments
	err = clSetKernelArg(m_viterbiKernel, 0, sizeof(cl_mem), (void*)&cmImg);
	err |= clSetKernelArg(m_viterbiKernel, 1, sizeof(cl_mem), (void*)&cmL);
	err |= clSetKernelArg(m_viterbiKernel, 2, sizeof(cl_mem), (void*)&cmLine_x);
	err |= clSetKernelArg(m_viterbiKernel, 3, sizeof(cl_mem), (void*)&cmV1);
	err |= clSetKernelArg(m_viterbiKernel, 4, sizeof(cl_mem), (void*)&cmV2);
	err |= clSetKernelArg(m_viterbiKernel, 5, sizeof(cl_int), (void*)&m_img_height);
	err |= clSetKernelArg(m_viterbiKernel, 6, sizeof(cl_int), (void*)&m_img_width);
	err |= clSetKernelArg(m_viterbiKernel, 7, sizeof(cl_int), (void*)&g_high);
	err |= clSetKernelArg(m_viterbiKernel, 8, sizeof(cl_int), (void*)&g_low);

	if (CL_SUCCESS != err)
	{
		return err; //
	}

	//to big buffer will fail with CL_MEM_OBJECT_ALLOCATION_FAILURE - have to process it with chunks
	//not all columns at the same time, call it couple of times
	size_t first_col = 0; 
	err = clEnqueueWriteBuffer(m_command_queue, cmLine_x, CL_FALSE, 0, sizeof(int) * m_img_width, line_x, 0, NULL, NULL);
	while (first_col < m_img_width && !err)
	{
		err = clSetKernelArg(m_viterbiKernel, 9, sizeof(cl_int), (void*)&first_col);
		err |= clEnqueueNDRangeKernel(m_command_queue, m_viterbiKernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);

		// Copy results from the memory buffer 
		err |= clEnqueueReadBuffer(m_command_queue, cmLine_x, CL_TRUE, 0,
			m_img_width * sizeof(int), line_x, 0, NULL, NULL);
		first_col += global_size;
	}
	
	line_x[m_img_width - 1] = line_x[m_img_width - 2];
	
	//realase resources
	err = clReleaseMemObject(cmLine_x);
	err = clReleaseMemObject(cmL);
	err = clReleaseMemObject(cmImg);
	err = clReleaseMemObject(cmV1);
	err = clReleaseMemObject(cmV2);
	return CL_SUCCESS;
}

void Viterbi::setImg(const unsigned char *img, size_t img_height, size_t img_width)
{
	m_img = img;
	m_img_width = img_width;
	m_img_height = img_height;
}

int Viterbi::launchViterbiMultiThread(std::vector<unsigned int>& line_x, int g_low, int g_high)
{
	uint32_t to_process = m_img_width - 1;
	uint32_t start_col = 0;
	uint32_t idx = 0;
	uint8_t num_of_threads = std::thread::hardware_concurrency();
	std::vector<unsigned int> line(num_of_threads);
	std::vector<std::future<unsigned int> > viterbiThreads(num_of_threads);
	while (to_process > 0)
	{
		uint8_t launched_threads = 0;
		for (uint8_t i = 0; i < num_of_threads; i++)
		{
			if (start_col < m_img_width - 1)
			{
				viterbiThreads[i] = (std::async(launch::async,
					&Viterbi::viterbiMultiThread, this, g_low, g_high, start_col));
				++start_col;
				++launched_threads;
				--to_process;
			}
			else
			{
				break;
			}
		}

		for (uint8_t i = 0; i < launched_threads; i++)
		{
			line_x[idx] = viterbiThreads[i].get();
			++idx;
		}
	}
	line_x[m_img_width - 1] = line_x[m_img_width - 2];
	return 0;
}

unsigned int Viterbi::viterbiMultiThread(int g_low, int g_high, unsigned int start_col)
{
	if (m_img == 0 && m_img_height > 0 && m_img_width > 0 && start_col < m_img_width)
	{
		return 1;
	}
	//allocate array for viterbi algorithm
	std::vector<int> L(m_img_height * m_img_width, 0);
	std::vector<uint32_t> V(m_img_height * m_img_width, 0);

	uint32_t max_val = 0;
	unsigned char pixel_value = 0;
	// init first column with zeros
	for (size_t m = 0; m < m_img_height; m++)
	{
		V[(m * m_img_width) + start_col] = 0;
	}
	for (size_t n = start_col; n < (m_img_width - 1); n++)
	{
		for (int j = 0; j < m_img_height; j++)
		{
			max_val = 0;
			for (int g = g_low; g <= g_high; g++)
			{
				if ((j + g) >(int)(m_img_height - 1))
				{
					break;
				}
				if (j + g < 0)
				{
					continue;
				}
				int curr_id = j + g;
				pixel_value = m_img[((curr_id)* m_img_width) + n];
				if ((pixel_value + V[(m_img_width * curr_id) + n]) > max_val)
				{
					max_val = pixel_value + V[(m_img_width * curr_id) + n];
					L[(j * m_img_width) + n] = g;
				}
			}
			V[(j * m_img_width) + (n + 1)] = max_val;
		}
	}

	uint32_t P_max = 0;
	uint32_t x_max = 0;
	//find biggest cost value in last column
	for (size_t j = 0; j < m_img_height; j++)
	{
		if (V[(j * m_img_width) + (m_img_width - 1)] > P_max)
		{
			P_max = V[(j * m_img_width) + (m_img_width - 1)];
			x_max = j;
		}
	}
	//backwards phase - retrace the path
	uint32_t x_n = x_max;
	for (size_t n = (m_img_width - 1); n > start_col; n--)
	{
		x_n = x_n + L[(x_n * m_img_width) + (n - 1)];
	}
	// save last pixel position
	return static_cast<unsigned int>(x_n);
}

double Viterbi::viterbiHybridCPU(std::vector<unsigned int> line_x, int g_lo, int g_high, uint32_t start_col, uint32_t end_col)
{
	return 0;
}

double Viterbi::viterbiHybridGPU(std::vector<unsigned int> line_x, int g_lo, int g_high, uint32_t start_col, uint32_t end_col)
{
	return 0;
}


int Viterbi::launchHybridViterbi(std::vector<unsigned int>& line_x, int g_low, int g_high)
{
	uint32_t start_col_CPU = 0, end_col_CPU = static_cast<uint32_t>(m_hybrid_rate.first * static_cast<double>(m_img_width));
	uint32_t start_col_GPU = end_col_CPU + 1, end_col_GPU = static_cast<uint32_t>(m_hybrid_rate.second * static_cast<double>(m_img_width));
	uint8_t num_of_threads = std::thread::hardware_concurrency();
	std::vector<unsigned int> line(num_of_threads);
	std::vector<std::future<unsigned int> > viterbiThreads(num_of_threads);

	std::future<double> cpu_thread = std::async(launch::async,
		&Viterbi::viterbiHybridCPU, this, std::ref(line_x), g_low, g_high, start_col_CPU, end_col_CPU);
	std::future<double> gpu_thread = std::async(launch::async,
		&Viterbi::viterbiHybridGPU, this, std::ref(line_x), g_low, g_high, start_col_GPU, end_col_GPU);

	double time_cpu = cpu_thread.get();
	double time_gpu = gpu_thread.get();

	line_x[m_img_width - 1] = line_x[m_img_width - 2];
	bool success = false;
	if (time_cpu > 0 && time_gpu > 0)
	{
		//calculate new rate
		if (!m_set_hybrid_rate)
		{
			double rate = static_cast<double>(time_cpu);
			m_hybrid_rate;// std::pair<double, double>(0.5, 0.5)
		}
		else
		{
			//handle check if image changed significantly its size in each dimension
			//if not use old rate, else set default
		}
	}
	return success;
}

