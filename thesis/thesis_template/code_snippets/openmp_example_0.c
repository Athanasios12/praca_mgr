#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void mp_test(void)
{
	int my_rank = omp_get_thread num();
	int thread_count = omp_get_num_threads();
	printf("Hello from thread %d of %d\n", my_rank, thread_count);
}

int main(int argc, char *argv[])
{
	int thread count = strtol(argv[1], NULL, 10);
	//parallel block start declaration
	# pragma omp parallel num_threads(thread count)
	{
		mp_test();
	}
	//ends with closing bracket
	return 0;
}