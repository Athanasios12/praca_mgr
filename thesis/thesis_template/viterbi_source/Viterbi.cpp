#include "Viterbi.h"
#include <iostream>
#include <mutex>
#include "Common_Tools.h"
#include <omp.h>

using namespace std;

Viterbi::Viterbi(const cl_command_queue & command_queue, const cl_context & context, cl_device_id device_id) :
	m_img(NULL),
	m_command_queue(command_queue),
	m_context(context),
	m_device_id(device_id),
	m_set_hybrid_rate(false),
	m_hybrid_rate(std::pair<double, double>(0.5, 0.5)),
	m_size_changed(false)
{
	m_initalized = loadAndBuildKernel();
}

Viterbi::~Viterbi()
{
	if (m_initalized)
	{
		clReleaseKernel(m_viterbiKernel);
		clReleaseKernel(m_viterbiHybridKernel);
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
	// Create OpenCL Kernels
	m_viterbiKernel = clCreateKernel(m_program, VITERBI_COLS_FUNCTION, &err);
	m_viterbiHybridKernel = clCreateKernel(m_program, VITERBI_GPU_FRAGMENT_FUNCTION, &err);
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
		(m_img_width * sizeof(int)) + (img_size * sizeof(unsigned char))) / double(1024 * 1024));
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
		first_col += global_size - 1;
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
	size_t old_height = m_img_height;
	size_t old_width = m_img_width;
	m_img_width = img_width;
	m_img_height = img_height;
	m_size_changed = false;
	//check if image size changed significantly
	if (m_set_hybrid_rate)
	{
		double tolerance = 0.1; //maybe get as parameter with default value - in function call
		double diff_h = (abs(static_cast<double>(m_img_height) - static_cast<double>(old_height))) / static_cast<double>(old_height);
		double diff_w = (abs(static_cast<double>(m_img_width) - static_cast<double>(old_width))) / static_cast<double>(old_width);
		if (diff_h > tolerance || diff_w > tolerance)
		{
			m_size_changed = true;
		}
	}
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

double Viterbi::viterbiHybridCPU(std::vector<unsigned int> &line_x, int g_low, int g_high, uint32_t start_col, uint32_t end_col)
{
	clock_t start = clock();
	uint32_t to_process = end_col - start_col;
	uint32_t idx = start_col;
	uint8_t num_of_threads = std::thread::hardware_concurrency() - 1;
	std::vector<unsigned int> line(num_of_threads);
	std::vector<std::future<unsigned int> > viterbiThreads(num_of_threads);
	while (to_process > 0)
	{
		uint8_t launched_threads = 0;
		for (uint8_t i = 0; i < num_of_threads; i++)
		{
			if (start_col < end_col)
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
	clock_t end = clock();
	return (double)(end - start); // maybe run first as test, without context changes - wait for one thread to finish, and compare times
}

double Viterbi::viterbiHybridGPU(unsigned int *line_x, int g_low, int g_high, uint32_t start_col, uint32_t end_col)
{
	clock_t start = clock();
	int err = 0;
	if (!m_initalized || m_img == NULL)
	{
		return -1;
	}
	size_t width = end_col - start_col;
	size_t img_size = (m_img_height * m_img_width);
	size_t L_size = m_img_height * width;
	size_t global_size = width;

	//check available memory
	cl_ulong dev_memory = 0;
	err = clGetDeviceInfo(m_device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &dev_memory, NULL);
	cl_ulong max_alloc = 0;
	err = clGetDeviceInfo(m_device_id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &max_alloc, NULL);
	int dev_mem = static_cast<int>(double(dev_memory) / double(1024 * 1024));//MB
	int max_buff_size = static_cast<int>(double(max_alloc) / double(1024 * 1024));
	int tot_mem = static_cast<int>(double((img_size * global_size * sizeof(float)) +
		(2 * m_img_height * global_size * sizeof(float)) +
		(m_img_width * sizeof(int)) + (img_size * sizeof(unsigned char))) / double(1024 * 1024));
	//handle not enough GPU memory
	if (max_buff_size < tot_mem)
	{
		int mem_multiple = (int)(tot_mem / max_buff_size);
		global_size = global_size / (mem_multiple + 1);
	}
	//create necessery opencl buffers
	cl_mem cmImg = clCreateBuffer(m_context, CL_MEM_READ_ONLY, sizeof(unsigned char) * img_size, NULL, &err);
	err = clEnqueueWriteBuffer(m_command_queue, cmImg, CL_FALSE, 0, sizeof(unsigned char) * img_size, m_img, 0, NULL, NULL);

	cl_mem cmLine_x = clCreateBuffer(m_context, CL_MEM_READ_WRITE, width * sizeof(int), NULL, &err);
	cl_mem cmV1 = clCreateBuffer(m_context, CL_MEM_READ_WRITE, m_img_height * global_size * sizeof(float), NULL, &err);
	cl_mem cmV2 = clCreateBuffer(m_context, CL_MEM_READ_WRITE, m_img_height * global_size * sizeof(float), NULL, &err);
	cl_mem cmL = clCreateBuffer(m_context, CL_MEM_READ_WRITE, L_size * global_size * sizeof(float), NULL, &err);

	//set kernel arguments
	err = clSetKernelArg(m_viterbiHybridKernel, 0, sizeof(cl_mem), (void*)&cmImg);
	err |= clSetKernelArg(m_viterbiHybridKernel, 1, sizeof(cl_mem), (void*)&cmL);
	err |= clSetKernelArg(m_viterbiHybridKernel, 2, sizeof(cl_mem), (void*)&cmLine_x);
	err |= clSetKernelArg(m_viterbiHybridKernel, 3, sizeof(cl_mem), (void*)&cmV1);
	err |= clSetKernelArg(m_viterbiHybridKernel, 4, sizeof(cl_mem), (void*)&cmV2);
	err |= clSetKernelArg(m_viterbiHybridKernel, 5, sizeof(cl_int), (void*)&m_img_height);
	err |= clSetKernelArg(m_viterbiHybridKernel, 6, sizeof(cl_int), (void*)&m_img_width);
	err |= clSetKernelArg(m_viterbiHybridKernel, 7, sizeof(cl_int), (void*)&width);
	err |= clSetKernelArg(m_viterbiHybridKernel, 8, sizeof(cl_int), (void*)&g_high);
	err |= clSetKernelArg(m_viterbiHybridKernel, 9, sizeof(cl_int), (void*)&g_low);
	err |= clSetKernelArg(m_viterbiHybridKernel, 11, sizeof(cl_int), (void*)&start_col);
	if (CL_SUCCESS != err)
	{
		return -1;
	}
	size_t first_col_linex = 0;
	err = clEnqueueWriteBuffer(m_command_queue, cmLine_x, CL_FALSE, 0, sizeof(int) * width, line_x, 0, NULL, NULL);
	while ((first_col_linex + start_col) <= end_col && !err)
	{
		err = clSetKernelArg(m_viterbiHybridKernel, 10, sizeof(cl_int), (void*)&first_col_linex);
		err |= clEnqueueNDRangeKernel(m_command_queue, m_viterbiHybridKernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);

		// Copy results from the memory buffer
		err |= clEnqueueReadBuffer(m_command_queue, cmLine_x, CL_TRUE, 0,
			width * sizeof(int), line_x, 0, NULL, NULL);
		first_col_linex += global_size - 1;
	}
	line_x[width] = line_x[width - 1];

	//realase resources
	err = clReleaseMemObject(cmLine_x);
	err = clReleaseMemObject(cmL);
	err = clReleaseMemObject(cmImg);
	err = clReleaseMemObject(cmV1);
	err = clReleaseMemObject(cmV2);
	clock_t end = clock();
	return (double)(end - start);
}

bool Viterbi::launchHybridViterbi(std::vector<unsigned int>& line_x, int g_low, int g_high)
{
	if (m_set_hybrid_rate && m_size_changed)
	{
		//set default rate, img size changed too much
		m_set_hybrid_rate = false;
		m_hybrid_rate = std::make_pair(0.5, 0.5);
	}
	uint32_t start_col_CPU = 0, end_col_CPU = static_cast<uint32_t>(m_hybrid_rate.first * static_cast<double>(m_img_width));
	uint32_t start_col_GPU = end_col_CPU, end_col_GPU = m_img_width - 1;
	uint8_t num_of_threads = std::thread::hardware_concurrency();
	std::vector<unsigned int> line(num_of_threads);
	std::vector<std::future<unsigned int> > viterbiThreads(num_of_threads);

	std::future<double> cpu_thread = std::async(launch::async, &Viterbi::viterbiHybridCPU,
		this, std::ref(line_x), g_low, g_high, start_col_CPU, end_col_CPU);
	std::unique_ptr<unsigned int> line_gpu(new unsigned int[end_col_GPU - start_col_GPU]);
	std::future<double> gpu_thread = std::async(launch::async, &Viterbi::viterbiHybridGPU,
		this, &line_x[start_col_GPU], g_low, g_high, start_col_GPU, end_col_GPU);

	double time_gpu = gpu_thread.get();
	double time_cpu = cpu_thread.get();
	//proof that cpu doesnt scale the same as the gpu - do 2 less tasks, still takes almost the same time to complete
	// prepare posix, openmp alternative for this bullshit - maybe will work better, check on different
	//cpu. In different case, does not make sense for larger images, gpu too fast, and cpu doesnt keep up
	print("Thread combo 1 : CPU time = " << time_cpu << "\t GPU time = " << time_gpu);
	print("CPU GPU thread combo 1 time : " << time_gpu + time_cpu);
	bool success = false;
	if (time_cpu > 0 && time_gpu > 0)
	{
		//calculate new rate
		if (!m_set_hybrid_rate)
		{
			double rate = time_cpu / (time_cpu + time_gpu);
			m_hybrid_rate = std::make_pair(1 - rate, rate);
			m_set_hybrid_rate = true;
		}
		success = true;
	}
	return success;
}

bool Viterbi::viterbiOpenMP(std::vector<unsigned int> &line_x, int g_low, int g_high)
{
	if (m_img == 0 && m_img_height > 0 && m_img_width > 0)
	{
		return false;
	}
	#pragma omp parallel
	{
		std::vector<uint32_t> L(m_img_height * m_img_width, 0);
		std::vector<uint32_t> V(m_img_height * m_img_width, 0);
		unsigned char pixel_value = 0;
		uint32_t P_max = 0;
		uint32_t x_max = 0;
		std::vector<uint32_t> x_cord(m_img_width, 0);
		uint32_t max_val = 0;
		#pragma omp for
		for (int i = 0; i < (m_img_width - 1); i++)
		{
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
			x_cord[(m_img_width - 1)] = x_max;
			for (size_t n = (m_img_width - 1); n > i; n--)
			{
				x_cord[n - 1] = x_cord[n] + L[(x_cord[n] * m_img_width) + (n - 1)];
			}
			// save only last pixel position
			line_x[i] = x_cord[i];
			P_max = 0;
			x_max = 0;
		}
	}
	line_x[m_img_width - 1] = line_x[m_img_width - 2];
	return true;
}

bool Viterbi::launchHybridViterbiOpenMP(std::vector<unsigned int> &line_x, int g_low, int g_high)
{
	if (m_set_hybrid_rate && m_size_changed)
	{
		m_set_hybrid_rate = false;
		m_hybrid_rate = std::make_pair(0.5, 0.5);
	}
	uint32_t start_col_CPU = 0;
	uint32_t end_col_CPU = static_cast<uint32_t>(m_hybrid_rate.first * static_cast<double>(m_img_width));
	uint32_t start_col_GPU = end_col_CPU;
	uint32_t end_col_GPU = m_img_width - 1;

	double time_gpu = 0;
	double time_cpu = 0;

	#pragma omp parallel num_threads(2)
	{
		auto thread_id = omp_get_thread_num();
		if (thread_id == 0)
		{
			time_cpu = viterbiHybridOpenMP_CPU(line_x, g_low, g_high, start_col_CPU, end_col_CPU);
		}
		else
		{
			time_gpu = viterbiHybridGPU(&line_x[start_col_GPU], g_low, g_high, start_col_GPU, end_col_GPU);
		}
	}
	//double tot_time = time_gpu + time_cpu;
	bool success = false;
	if (time_cpu > 0 && time_gpu > 0)
	{
		//calculate new rate
		if (!m_set_hybrid_rate)
		{
			double rate = (time_cpu / (time_cpu + time_gpu));
			m_hybrid_rate = std::make_pair(1 - rate, rate);
			m_set_hybrid_rate = true;
		}
		success = true;
	}
	return success;
}

double Viterbi::viterbiHybridOpenMP_CPU(std::vector<unsigned int> &line_x, int g_low, int g_high, uint32_t start_col, uint32_t end_col)
{
	clock_t start = clock();
	if (m_img == 0 && m_img_height > 0 && m_img_width > 0)
	{
		return false;
	}
	#pragma omp parallel
	{
		std::vector<uint32_t> L(m_img_height * m_img_width, 0);
		std::vector<uint32_t> V(m_img_height * m_img_width, 0);
		unsigned char pixel_value = 0;
		uint32_t P_max = 0;
		uint32_t x_max = 0;
		std::vector<uint32_t> x_cord(m_img_width, 0);
		uint32_t max_val = 0;
		#pragma omp for
		for (int i = start_col; i <= end_col; i++)
		{
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
			x_cord[(m_img_width - 1)] = x_max;
			for (size_t n = (m_img_width - 1); n > i; n--)
			{
				x_cord[n - 1] = x_cord[n] + L[(x_cord[n] * m_img_width) + (n - 1)];
			}
			// save only last pixel position
			line_x[i] = x_cord[i];
			P_max = 0;
			x_max = 0;
		}
	}
	clock_t end = clock();
	return static_cast<double>(end - start);
}