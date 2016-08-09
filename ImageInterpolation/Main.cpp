/**********************************************************************************/
/* The MIT License(MIT)                                                           */
/*                                                                                */
/* Copyright(c) 2016-2016 Matthieu Vendeville                                     */
/*                                                                                */
/* Permission is hereby granted, free of charge, to any person obtaining a copy   */
/* of this software and associated documentation files(the "Software"), to deal   */
/* in the Software without restriction, including without limitation the rights   */
/* to use, copy, modify, merge, publish, distribute, sublicense, and / or sell    */
/* copies of the Software, and to permit persons to whom the Software is          */
/* furnished to do so, subject to the following conditions :                      */
/*                                                                                */
/* The above copyright notice and this permission notice shall be included in all */
/* copies or substantial portions of the Software.                                */
/*                                                                                */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     */
/* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,       */
/* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE     */
/* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         */
/* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  */
/* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  */
/* SOFTWARE.                                                                      */
/**********************************************************************************/

#include <stdio.h>
#include "ImCpu.h"
#include "ImGpu.h"
#include "cuda_profiler_api.h"
#include <time.h>
#include <iostream>
using namespace std;

int main(int argc, char** argv) 
{
	Im *Im1,*Im2;
	cudaError_t cudaStatus;
	clock_t begin_time, end_time = 0;
	int i;
	
	char *device_type = "gpu";
	char *interpolation_type = "nn";
	int iterations = 10;
	int new_width = 8000;
	int new_height = 4000;


	//double ff = atof(argv[1]);
	//std::cout << "float number: " << ff << '\n';
	//int a = ff + 0.5;
	//std::cout << "Integer number: " << a << '\n';
	//a = ff - 0.5;
	//std::cout << "Integer number: " << a << '\n';
	//return 0;




	if (argc > 1 ){
		device_type = argv[1];
		iterations = atoi(argv[2]);
		interpolation_type = argv[3];
		std::cout << "Using device: " << device_type <<'\n';
		std::cout << "Nb iterations: " << iterations << '\n';
		std::cout << "Interplation types: " << interpolation_type << '\n';
	}

	//
	// Initialise GPU
	//
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}

	//
	// Init instance, depending on process happening on GPU or CPU
	//
	if (strcmp(device_type, "cpu") == 0){
		Im1 = new ImCpu("512x512x8x1_lena.dat");
		std::cout << "Creating Imcpu instance" << '\n';
	}
	else{
		Im1 = new ImGpu("512x512x8x1_lena.dat");
		std::cout << "Creating Imgpu instance" << '\n';
	}

	//
	// Perform and profile interpolation x times 
	//
	begin_time = clock();
	for (i = 0; i < iterations; i++){
		Im2 = Im1->clone();
		if (strcmp(interpolation_type, "nn") == 0)
		{
			Im2->InterpolateNN(8000, 4000);
		}
		else
		{
			Im2->InterpolateBilinear(8000, 4000);
		}
		delete(Im2);
	}
	end_time = clock() - begin_time;

	std::cout << float(end_time) / CLOCKS_PER_SEC << '\n';

	//
	// Save processed imaged
	//
	if (strcmp(interpolation_type, "nn") == 0)
	{
		Im1->InterpolateNN(8000, 4000);
	}
	else
	{
		Im1->InterpolateBilinear(8000, 4000);
	}

	Im1->Save2RawFile("popo.dat");

	exit(0);
	
}

