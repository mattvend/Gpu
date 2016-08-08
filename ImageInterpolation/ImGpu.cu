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
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

__global__ void KernelInterpolateNN(void *pxl, void *new_pxl, float WidthScaleFactor, float HeightScaleFactor, unsigned short new_width, unsigned short width)
{
	unsigned short  XRounded, YRounded;
	
	// X and Y are pixels coordinates in destination image
	unsigned short X = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned short Y = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	// XRounded and YRounded are coordinates of the nearest neighbor in the original image */
//	XRounded = (unsigned short)floor((float)X*WidthScaleFactor);
//	YRounded = (unsigned short)floor((float)Y*HeightScaleFactor);

	XRounded = (unsigned short)__fmul_rd((float)X, WidthScaleFactor);
	YRounded = (unsigned short)__fmul_rd((float)Y, HeightScaleFactor);

	*((char*)new_pxl + X + Y*new_width) = *((char*)pxl + XRounded + YRounded*width);

}

#define ImPxl(IM,X,Y,W)     *((unsigned char*)IM + (X) + (Y)*W)

__global__ void KernelInterpolateBilinear(void *pxl, void *new_pxl, unsigned short new_width, unsigned short width, unsigned short new_height, unsigned short height, float WidthScaleFactor, float HeightScaleFactor)
{
	unsigned short	X = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	unsigned short	Y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
	unsigned short	Xp1, Xp2, Xp3, Xp4;
	unsigned short	Yp1, Yp2, Yp3, Yp4;
	unsigned short	Integer;

	/* Compute scaling factor for each dimension */
//	float HeightScaleFactor = ((float)height / (float)new_height);
//	float WidthScaleFactor = ((float)width / (float)new_width);

	double xdest, ydest;
	double alphax, alphay;

	/* Compute pixel intensity in destination image */

	/*
	* xdest and ydest are coordinates of destination pixel in the original image
	*/
//	xdest = (float)(X + .5)*WidthScaleFactor;
//	ydest = (float)(Y + .5)*HeightScaleFactor;

	xdest = __fmul_rd((float)(X + .5), WidthScaleFactor);
	ydest = __fmul_rd((float)(Y + .5), HeightScaleFactor);

	//  printf("Xdest=%f Ydest=%f ",xdest,ydest);

	/* Processing pixels in the top left corner */
	if ((xdest < 0.5) && (ydest < 0.5))
	{
		ImPxl(new_pxl, X, Y, new_width) = ImPxl(pxl, 0, 0, width);
	}

	/* Processing pixels in the top center */
	if ((xdest > 0.5) && (ydest < 0.5) && (xdest < (width - 1 + 0.5)))
	{
		/* Compute Alpha x value used to perform interpolation */

		Integer = (unsigned short)(xdest - 0.5);
		alphax = (float)((xdest - 0.5) - Integer);
		Xp1 = Integer;
		Xp2 = Xp1 + 1;

		// (1 - t)*v0 + t*v1; // fma(t, v1, fma(-t, v0, v0))
		
		/* Perform bilinear interpolation */
		ImPxl(new_pxl, X, Y, new_width) = (unsigned char)((1 - alphax)*ImPxl(pxl, Xp1, 0, width) + alphax*ImPxl(pxl, Xp2, 0, width));
		// ImPxl(new_pxl, X, Y, new_width) = (unsigned char)fma(alphax, ImPxl(pxl, Xp2, 0, width), fma(-alphax, ImPxl(pxl, Xp1, 0, width), ImPxl(pxl, Xp1, 0, width)));

	}

	/* Processing pixels in the top right corner */
	if ((ydest < 0.5) && (xdest >(width - 1 + 0.5)))
	{
		/* Taking last pixel of the first row */
		ImPxl(new_pxl, X, Y, new_width) = ImPxl(pxl, width - 1, 0, width);
	}

	/* Processing pixels in left side, center */
	if ((xdest < 0.5) && (ydest > 0.5) && (ydest < (height - 1 + 0.5)))
	{
		/* Compute Alpha y value used to perform interpolation */
		Integer = (unsigned short)(ydest - 0.5);
		alphay = (float)((ydest - 0.5) - Integer);

		Yp1 = Integer;
		Yp3 = Yp1 + 1;

		/* Perform bilinear interpolation */
		ImPxl(new_pxl, X, Y, new_width) = (unsigned char)((1 - alphay)*ImPxl(pxl, 0, Yp1, width) + alphay*ImPxl(pxl, 0, Yp3, width));
	}

	/* Processing pixels in the center */
	if ((xdest > 0.5) && (ydest > 0.5) && (xdest < (width - 1 + 0.5)) && (ydest < (height - 1 + 0.5)))
	{
		/*
		* Compute Alpha x and Alpha y values used to perform interpolation
		*/

		Integer = (unsigned short)(xdest - 0.5);
		alphax = (float)((xdest - 0.5) - Integer);
		Xp1 = Xp3 = Integer;
		Xp2 = Xp4 = Xp1 + 1;

		Integer = (unsigned short)(ydest - 0.5);
		alphay = (float)((ydest - 0.5) - Integer);

		Yp1 = Yp2 = Integer;
		Yp3 = Yp4 = Yp1 + 1;

		/* Perform bilinear interpolation */
		ImPxl(new_pxl, X, Y, new_width) = (unsigned char)((1 - alphax)*(1 - alphay)*ImPxl(pxl, Xp1, Yp1, width) + alphax*(1 - alphay)*ImPxl(pxl, Xp2, Yp2, width) + (1 - alphax)*alphay*ImPxl(pxl, Xp3, Yp3, width) + alphay*alphax*ImPxl(pxl, Xp4, Yp4, width));
	}

	/* Processing pixels in right side, center */
	if ((xdest > (width - 1 + 0.5)) && (ydest > 0.5) && (ydest < (height - 1 + 0.5)))
	{
		/*
		* Compute Alpha y values used to perform interpolation
		*/
		Integer = (unsigned short)(ydest - 0.5);
		alphay = (float)((ydest - 0.5) - Integer);

		Yp1 = Yp2 = Integer;
		Yp3 = Yp4 = Yp1 + 1;

		/* Perform bilinear interpolation */
		ImPxl(new_pxl, X, Y, new_width) = (unsigned char)((1 - alphay)*ImPxl(pxl, (width - 1), Yp1, width) + alphay*ImPxl(pxl, (width - 1), Yp3, width));
	}

	/* Processing pixels in the lower left corner */
	if ((xdest < 0.5) && (ydest >(height - 1 + 0.5)))
	{
		ImPxl(new_pxl, X, Y, new_width) = ImPxl(pxl, 0, height - 1, width);
	}

	/* Processing pixels in bottom , center */
	if ((xdest > 0.5) && (xdest < (width - 1 + 0.5)) && (ydest >(height - 1 + 0.5)))
	{
		/*
		* Compute Alpha x values used to perform interpolation
		*/
		Integer = (unsigned short)(xdest - 0.5);
		alphax = (float)((xdest - 0.5) - Integer);
		Xp1 = Integer;
		Xp2 = Xp1 + 1;

		/* Perform bilinear interpolation */
		ImPxl(new_pxl, X, Y, new_width) = (unsigned char)((1 - alphax)*ImPxl(pxl, Xp1, height - 1, width) + alphax*ImPxl(pxl, Xp2, height - 1, width));
	}

	/* Processing pixels in the lower right corner */
	if ((xdest > (width - 1 + 0.5)) && (ydest > (height - 1 + 0.5)))
	{
		ImPxl(new_pxl, X, Y, new_width) = ImPxl(pxl, width - 1, height - 1, width);
	}

	return;
}



void ImGpu::InterpolateNN(unsigned short new_width, unsigned short new_height)
{
	void *new_pxl;
	void *dev_new_pxl;
	void *dev_cur_pxl;

	cudaError_t cudaStatus;


	/* Compute scaling factor for each dimension */
	float HeightScaleFactor = ((float)height / (float)new_height);
	float WidthScaleFactor = ((float)width / (float)new_width);

	/* Allocate memory for the pixels on the CPU */
	if (8 == bpp)
	{
		new_pxl = new char[sizeof(char) * new_width *new_height *dimension];
	}
	else if (16 == bpp)
	{
		new_pxl = new unsigned short[sizeof(unsigned short) * new_width *new_height *dimension];
	}

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for the buffers of pixels on the GPU.
	cudaStatus = cudaMalloc((void**)&dev_new_pxl, new_width *new_height *dimension * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_cur_pxl, width *height *dimension * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_cur_pxl, pxl, width *height *dimension * sizeof(char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_new_pxl, new_pxl, new_width *new_height *dimension * sizeof(char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


	// Launch a kernel on the GPU with one thread for each element.
	{
		dim3 threadsPerBlock(16, 16);  // 64 threads
		dim3 numBlocks((new_width + threadsPerBlock.x - 1) / threadsPerBlock.x, (new_height + threadsPerBlock.y - 1) / threadsPerBlock.y);
		KernelInterpolateNN <<< numBlocks, threadsPerBlock >>> (dev_cur_pxl, dev_new_pxl, WidthScaleFactor, HeightScaleFactor, new_width, width);
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

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(new_pxl, dev_new_pxl, new_width *new_height *dimension * sizeof(char), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	delete(pxl);
	pxl = new_pxl;

	width = new_width;
	height = new_height;

Error:
	cudaFree(dev_new_pxl);
	cudaFree(dev_cur_pxl);

	return;
}

void ImGpu::InterpolateBilinear(unsigned short new_width, unsigned short new_height)
{
	void *new_pxl;
	void *dev_new_pxl;
	void *dev_cur_pxl;

	cudaError_t cudaStatus;

	/* Allocate memory for the pixels on the CPU */
	if (8 == bpp)
	{
		new_pxl = new char[sizeof(char) * new_width *new_height *dimension];
	}
	else if (16 == bpp)
	{
		new_pxl = new unsigned short[sizeof(unsigned short) * new_width *new_height *dimension];
	}

	// Allocate GPU buffers for the buffers of pixels on the GPU.
	cudaStatus = cudaMalloc((void**)&dev_new_pxl, new_width *new_height *dimension * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_cur_pxl, width *height *dimension * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_cur_pxl, pxl, width *height *dimension * sizeof(char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_new_pxl, new_pxl, new_width *new_height *dimension * sizeof(char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


	// Launch a kernel on the GPU with one thread for each element.
	{
		/* Compute scaling factor for each dimension */
		float HeightScaleFactor = ((float)height / (float)new_height);
		float WidthScaleFactor = ((float)width / (float)new_width);

		dim3 threadsPerBlock(8, 8);  // 64 threads
		dim3 numBlocks((new_width + threadsPerBlock.x - 1) / threadsPerBlock.x, (new_height + threadsPerBlock.y - 1) / threadsPerBlock.y);
		KernelInterpolateBilinear << < numBlocks, threadsPerBlock >> > (dev_cur_pxl, dev_new_pxl, new_width, width, new_height, height, WidthScaleFactor, HeightScaleFactor);
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

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(new_pxl, dev_new_pxl, new_width *new_height *dimension * sizeof(char), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	delete(pxl);
	pxl = new_pxl;

	width = new_width;
	height = new_height;

Error:
	cudaFree(dev_new_pxl);
	cudaFree(dev_cur_pxl);

	return;
}