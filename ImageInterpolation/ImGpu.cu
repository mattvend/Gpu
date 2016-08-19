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

#include "ImGpu.h"
#include <stdio.h>
#include <iostream>

__global__ void ComputeXDest(float *xdest, float WidthScaleFactor)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	xdest[idx] = (float)(idx + .5)*WidthScaleFactor;
}

__global__ void ComputeXDestBL(float WidthScaleFactor, unsigned short width, double *ax, unsigned short *Ix1, unsigned short *Ix2)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	float xdest = (float)(idx + .5)*WidthScaleFactor;

	if (xdest <= 0.5){
		ax[idx] = 1;
		Ix1[idx] = 0;
		Ix2[idx] = 0;
	}

	if ((xdest > 0.5) && (xdest < (width - 1 + 0.5)))
	{
		/* Compute Alpha x value used to perform interpolation */
		unsigned short Integer = (unsigned short)(xdest - 0.5);
		ax[idx] = (float)((xdest - 0.5) - Integer);
		Ix1[idx] = Integer;
		Ix2[idx] = Ix1[idx] + 1;
	}

	if (xdest >= (width - 1 + 0.5))
	{
		ax[idx] = 0;
		Ix1[idx] = width - 1;
		Ix2[idx] = width - 1;
	}
}

__global__ void ComputeYDest(float *ydest, float HeightScaleFactor)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	ydest[idx] = (float)(idx + .5)*HeightScaleFactor;
}

__global__ void ComputeYDestBL(float HeightScaleFactor, unsigned short heigth, double *ay, unsigned short *Iy1, unsigned short *Iy2)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	float ydest = (float)(idx + .5)*HeightScaleFactor;

	if (ydest <= 0.5){
		ay[idx] = 1;
		Iy1[idx] = 0;
		Iy2[idx] = 0;
	}

	if ((ydest > 0.5) && (ydest < (heigth - 1 + 0.5)))
	{
		/* Compute Alpha x value used to perform interpolation */
		unsigned short Integer = (unsigned short)(ydest - 0.5);
		ay[idx] = (float)((ydest - 0.5) - Integer);
		Iy1[idx] = Integer;
		Iy2[idx] = Iy1[idx] + 1;
	}

	if (ydest >= (heigth - 1 + 0.5))
	{
		ay[idx] = 0;
		Iy1[idx] = heigth - 1;
		Iy2[idx] = heigth - 1;
	}
}




__global__ void KernelInterpolateNN(void *pxl, void *new_pxl, float *xdest, float *ydest, unsigned short new_width, unsigned short width)
{
	unsigned short  XRounded, YRounded;

	// X and Y are pixels coordinates in destination image
	unsigned short X = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned short Y = (blockIdx.y * blockDim.y) + threadIdx.y;

	// XRounded and YRounded are coordinates of the nearest neighbor in the original image */
	XRounded = (unsigned short)xdest[X];
	YRounded = (unsigned short)ydest[Y];

	*((char*)new_pxl + X + Y*new_width) = *((char*)pxl + XRounded + YRounded*width);
}

#define ImPxl(IM,X,Y,W)     *((unsigned char*)IM + (X) + (Y)*W)

__global__ void KernelInterpolateBilinear(void *pxl, void *new_pxl, unsigned short new_width, unsigned short width, double *ax, unsigned short *Ix1, unsigned short *Ix2, double *ay, unsigned short *Iy1, unsigned short *Iy2)
{
	unsigned short  X = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned short  Y = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	/* Perform bilinear interpolation */
	ImPxl(new_pxl, X, Y, new_width) = (unsigned char)((1 - ax[X])*(1 - ay[Y])* ImPxl(pxl, Ix1[X], Iy1[Y], width) + \
														  ax[X] * (1 - ay[Y])* ImPxl(pxl, Ix2[X], Iy1[Y], width) + \
														   (1 - ax[X])*ay[Y] * ImPxl(pxl, Ix1[X], Iy2[Y], width) + \
															   ay[Y] * ax[X] * ImPxl(pxl, Ix2[X], Iy2[Y], width));
	return;
}



void ImGpu::InterpolateNN(unsigned short new_width, unsigned short new_height)
{
	void *dev_new_pxl;
	cudaError_t cudaStatus;
	float *xdest, *ydest;

	/* Compute scaling factor for each dimension */
	float HeightScaleFactor = ((float)height / (float)new_height);
	float WidthScaleFactor = ((float)width / (float)new_width);

	// Allocate GPU buffers for the buffers of pixels on the GPU.
	cudaStatus = cudaMalloc((void**)&dev_new_pxl, new_width *new_height *dimension * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&xdest, new_width * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&ydest, new_height * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	// Using Streams and events API to synchronize kernel calls
	{
		dim3 threadsPerBlock(16, 16);  // 64 threads
		dim3 numBlocks((new_width + threadsPerBlock.x - 1) / threadsPerBlock.x, (new_height + threadsPerBlock.y - 1) / threadsPerBlock.y);
		cudaStream_t streamA, streamB, streamC;;
		cudaEvent_t event, event2;

		cudaStreamCreate(&streamA);
		cudaStreamCreate(&streamB);
		cudaStreamCreate(&streamC);

		cudaEventCreate(&event);
		cudaEventCreate(&event2);


		ComputeXDest << < (new_width + 96 - 1) / 96, 96, 0, streamA >> > (xdest, WidthScaleFactor);
		cudaEventRecord(event, streamA);


		ComputeYDest << < (new_height + 96 - 1) / 96, 96,0, streamB >> > (ydest, HeightScaleFactor);
		cudaEventRecord(event2, streamB);

		// Do not call KernelInterpolateNN until computation needed has been done in ComputeXDest and ComputeYDest
		// Ensuring correct stream synchro with cudaStreamWaitEvent
		
		cudaStreamWaitEvent(streamC, event, 0);
		cudaStreamWaitEvent(streamC, event2, 0);

		KernelInterpolateNN << < numBlocks, threadsPerBlock,0, streamC >> > (dev_pxl, dev_new_pxl, xdest, ydest, new_width, width);
	}

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Free all resources
	cudaFree(dev_pxl);
	cudaFree(xdest);
	cudaFree(ydest);

	dev_pxl = dev_new_pxl;

	width = new_width;
	height = new_height;

	return;
Error:
	cudaFree(dev_new_pxl);
	cudaFree(xdest);
	cudaFree(ydest);
}

void ImGpu::InterpolateBilinear(unsigned short new_width, unsigned short new_height)
{
	void *dev_new_pxl;
	cudaError_t cudaStatus;
	double *ax, *ay;
	unsigned short *Ix1, *Ix2, *Iy1, *Iy2;

	/* Compute scaling factor for each dimension */
	float HeightScaleFactor = ((float)height / (float)new_height);
	float WidthScaleFactor = ((float)width / (float)new_width);

	// Allocate GPU buffers for the buffers of pixels on the GPU.
	cudaStatus = cudaMalloc((void**)&dev_new_pxl, new_width *new_height *dimension * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&Ix1, new_width * sizeof(unsigned short));
	cudaStatus = cudaMalloc((void**)&Ix2, new_width * sizeof(unsigned short));
	cudaStatus = cudaMalloc((void**)&Iy1, new_height * sizeof(unsigned short));
	cudaStatus = cudaMalloc((void**)&Iy2, new_height * sizeof(unsigned short));
	
	cudaStatus = cudaMalloc((void**)&ax, new_width * sizeof(double));
	cudaStatus = cudaMalloc((void**)&ay, new_height * sizeof(double));

	// Launch a kernel on the GPU with one thread for each element.
	{
		dim3 threadsPerBlock(16, 16);  // 64 threads
		dim3 numBlocks((new_width + threadsPerBlock.x - 1) / threadsPerBlock.x, (new_height + threadsPerBlock.y - 1) / threadsPerBlock.y);
		cudaStream_t streamA, streamB, streamC;;
		cudaEvent_t event, event2;

		cudaStreamCreate(&streamA);
		cudaStreamCreate(&streamB);
		cudaStreamCreate(&streamC);

		cudaEventCreate(&event);
		cudaEventCreate(&event2);

		ComputeXDestBL << < (new_width + 96 - 1) / 96, 96, 0, streamA >> > (WidthScaleFactor, width, ax, Ix1, Ix2);
		cudaEventRecord(event, streamA);

		ComputeYDestBL << < (new_height + 96 - 1) / 96, 96, 0, streamB >> > ( HeightScaleFactor, height, ay, Iy1, Iy2);
		cudaEventRecord(event2, streamB);

		// Do not call KernelInterpolateNN until computation needed has been done in ComputeXDest and ComputeYDest
		// Ensuring correct stream synchro with cudaStreamWaitEvent
		cudaStreamWaitEvent(streamC, event, 0);
		cudaStreamWaitEvent(streamC, event2, 0);

		KernelInterpolateBilinear << < numBlocks, threadsPerBlock, 0, streamC >> > (dev_pxl, dev_new_pxl, new_width, width, ax, Ix1, Ix2, ay, Iy1, Iy2);
	}

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Free all resources
	cudaFree(dev_pxl);

	cudaFree(Ix1);
	cudaFree(Ix2);
	cudaFree(Iy1);
	cudaFree(Iy2);

	cudaFree(ax);
	cudaFree(ay);
	
	dev_pxl = dev_new_pxl;

	width = new_width;
	height = new_height;

	return;

Error:
	cudaFree(dev_new_pxl);
	
	cudaFree(Ix1);
	cudaFree(Ix2);
	cudaFree(Iy1);
	cudaFree(Iy2);

	cudaFree(ax);
	cudaFree(ay);
}