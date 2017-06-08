__kernel void viterbi_function(__global const unsigned char *img,
								__global float *L, 
								__global int *line_x, 
								__global float* V_1,
								__global float* V_2,
								int img_height,
								int img_width, 
								int g_high,
								int g_low,
								int first_col)
{
	int start_column = get_global_id(0); //start_column - column from global buffer, but data can be bigger than global size
	long L_id = img_height * img_width * start_column; 
	int V_id = img_height *start_column;
	int global_size = get_global_size(0);

	float P_max = 0;
	float x_max = 0;
	float max_val = 0;
	float pixel_value = 0;
	__global float *temp_buffer; // maybe may need to be passed with size as argument and set with clSetKernelArgs
	__global float *V_old = &V_1[V_id];
	__global float *V_new = &V_2[V_id];
	// init first column with zeros
	int n = 0;
	int x_n = 0;
	for (int m = 0; m < img_height; m++)
	{
		V_old[m] = 0;
	}
	for (n = start_column; n < (global_size - 1) && ((n + first_col) < (img_width - 1)); n++)
	{
		for (int j = 0; j < img_height; j++)
		{
			max_val = 0;
			for (int g = g_low; g <= g_high; g++)
			{
				if ((j + g) > (img_height - 1))
				{
					break;
				}
				if ((j + g) < 0)
				{
					continue;
				}
				pixel_value = img[((j + g) * img_width) + (n + first_col)];
				if ((pixel_value + V_old[j + g]) > max_val)
				{
					max_val = pixel_value + V_old[j + g];
					L[L_id + (j * img_width) + n] = g;
				}
			}
			V_new[j] = max_val;
		}
		temp_buffer = &V_old[0]; // have to do it or both pointers will have same adress
		V_old = &V_new[0];
		V_new = &temp_buffer[0];
	}
	//find biggest cost value in last column
	for (int j = 0; j < img_height; j++)
	{
		if (V_old[j] > P_max)
		{
			P_max = V_old[j];
			x_max = j;
		}
	}
	//backwards phase - retrace the path
	x_n = x_max;
	for (; n > start_column; n--)
	{
		x_n = x_n + L[L_id + (x_n * img_width) + (n - 1)]; //L[L_id][row][column]
	}
	// save only last pixel position
	line_x[start_column + first_col] = x_n;
}
