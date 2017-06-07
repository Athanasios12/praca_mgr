__kernel void viterbi_forward( __global const unsigned char * img, __global float *L, __global float *V, int img_height, int img_width, int column, int g_low, int g_high)
{
	int row = get_global_id(0);
	if (row >= img_height)
	{
		return;
	}
	float max_val = 0;
	unsigned char pixel_value = 0;
	for (int g = g_low; g <= g_high; g++)
	{
		if ((row + g) > (img_height - 1))
		{
			break;
		}
		if ((row + g) < 0)
		{
			continue;
		}
		pixel_value = img[(img_width * (row + g)) + column];
		if ((pixel_value + V[((row + g) * img_width) + (column)]) > max_val)
		{
			max_val = pixel_value + V[((row + g) * img_width) + (column)];
			L[(row * img_width) + column] = g;
		}
	}
	V[(row * img_width) + (column + 1)] = max_val;
	//add blockade here to wait for all rows to get processed, after
	//do the same trick with Vnew and Vold instead of calculating
	//and storing V for every column not calculate onl for one column and call it again,
	//instead call it for every column in range from <start_col; img_width - 2>
	//After finishing all columns outside of kernel backtrack and save the solution, then call again for
	//next start_col until start_col = img_width - 1;
}

__kernel void initV(__global float *V, int img_height, int img_width, int start_column)
{
	int row = get_global_id(0);
	if (row >= img_height)
	{
		return;
	}
	V[(row * img_width) + start_column] = 0;	
}

__kernel void viterbi_function(__global const unsigned char *img,
								__global float *L, 
								__global int *line_x, 
								__global float* V_1,
								__global float* V_2,
								__global int *x_cord,
								int img_height,
								int img_width, 
								int g_high,
								int g_low,
								int first_col)
{
	int start_column = get_global_id(0);
	long L_id = img_height * img_width * start_column; 
	int V_id = img_height *start_column;
	int global_size = get_global_size(0);

	float P_max = 0;
	float x_max = 0;
	float max_val = 0;
	float pixel_value = 0;
	int n = 0;
	
	//buffer for pointer adress change
	__global float *temp_buffer;
	
	//pointers emulating V accumulation matrix
	__global float *V_old = &V_1[V_id];
	__global float *V_new = &V_2[V_id];
	
	// init first column with zeros	
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
	x_cord[n] = x_max;
	for (; n > start_column; n--)
	{
		x_cord[n - 1] = x_cord[n] + L[L_id + (x_cord[n] * img_width) + (n - 1)]; //L[L_id][row][column]
	}
	// save only last pixel position
	line_x[start_column + first_col] = x_cord[start_column];
}
