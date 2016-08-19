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

#include "ImCpu.h"
#include <stdio.h>
#include <iostream>


ImCpu::ImCpu(unsigned short width, unsigned short height, unsigned short bpp, unsigned short dimension)
{
	width = width;
	height = height;
	bpp = bpp;
	dimension = 1;
	pxl = 0;

	/* Allocate memory for the pixels */
	if (8 == bpp)
	{
		pxl = new char[sizeof(char) * width *height *dimension];
		memset(pxl, 255, sizeof(char) * width *height *dimension);
	}
	else if (16 == bpp)
	{
		pxl = new unsigned short[sizeof(unsigned short) * width *height *dimension];
		memset(pxl, 255, sizeof(unsigned short) * width *height *dimension);
	}
}

ImCpu::ImCpu(const ImCpu &obj)// :Im(obj)
{
	width = obj.width;
	height = obj.height;
	bpp = obj.bpp;
	dimension = 1;
	pxl = 0;

	/* Allocate memory for the pixels */
	if (8 == bpp)
	{
		pxl = new char[sizeof(char) * width *height *dimension];
	}
	else if (16 == bpp)
	{
		pxl = new unsigned short[sizeof(unsigned short) * width *height *dimension];
	}

	memcpy(pxl, obj.pxl, sizeof(char) * width *height *dimension);
}

ImCpu::ImCpu(const char* filename)
{
	FILE *fp = 0;
	int t1, t2, t3, t4;

	sscanf_s(filename, "%dx%dx%dx%d_", &t1, &t2, &t3, &t4);

	width = t1;
	height = t2;
	bpp = t3;
	dimension = t4;

	pxl = 0;

	/* Allocate memory for the pixels */
	if (8 == bpp)
	{
		pxl = new unsigned char[sizeof(unsigned char) * width *height *dimension];
	}
	else if (16 == bpp)
	{
		pxl = new unsigned short[sizeof(unsigned short) * width *height *dimension];
	}

	/*
	* Open the file to read the pixels
	*/
	fopen_s(&fp, filename, "rb"); /* open for reading */

	if (0 != fp){
		std::fread(pxl, sizeof(unsigned char), width*height*dimension, fp);
		//fread( pxl,sizeof(char),height*dimension,fp);
		fclose(fp); /* close the file */
	}
}

ImCpu::~ImCpu(void)
{
	delete(pxl);
}


void ImCpu::Save2RawFile(const char* filename)
{
	//char name[256];
	FILE *fp;

	sprintf_s(raw_file_name, "%dx%dx%dx%d_%s", width, height, bpp, dimension, filename);

	fopen_s(&fp, raw_file_name, "wb"); /* open for writing */

	if (8 == bpp)
	{
		fwrite(pxl, sizeof(char), width *height *dimension, fp);
	}
	else if (16 == bpp)
	{
		fwrite(pxl, sizeof(unsigned short), width *height *dimension, fp);
	}

	fclose(fp); /* close the file before ending program */

	return;
}

void ImCpu::PrintRawFileName()
{
	std::cout << raw_file_name << '\n';
}

void ImCpu::InterpolateNN(unsigned short new_width, unsigned short new_height)
{
	unsigned short     X, Y;
	unsigned short     XRounded, YRounded;
	void *new_pxl;

	/* Compute scaling factor for each dimension */
	float HeightScaleFactor = ((float)height / (float)new_height);
	float WidthScaleFactor = ((float)width / (float)new_width);
	float *xdest, *ydest;

	/* Allocate memory for the pixels */
	if (8 == bpp)
	{
		new_pxl = new char[sizeof(char) * new_width *new_height *dimension];
		xdest = new float[new_width];
		ydest = new float[new_height];
	}
	else if (16 == bpp)
	{
		new_pxl = new unsigned short[sizeof(unsigned short) * new_width *new_height *dimension];
	}

	for (Y = 0; Y < new_height; Y++)
	{
		ydest[Y] = (float)(Y + .5)*HeightScaleFactor;
	}
	
	for (X = 0; X < new_width; X++)
	{
		xdest[X] = (float)(X + .5)*WidthScaleFactor;
	}
	
	/* Compute pixel intensity in destination image */
	for (Y = 0; Y < new_height; Y++)
	{
		for (X = 0; X < new_width; X++)
		{
			/* xdest and ydest are coordinates of destination pixel in the original image */
			XRounded = (unsigned short)xdest[X];
			YRounded = (unsigned short)ydest[Y];

			*((char*)new_pxl + X + Y*new_width) = *((char*)pxl + XRounded + YRounded*width);
		}
	}

	delete(pxl);
	pxl = new_pxl;

	width = new_width;
	height = new_height;

	return;
}

#define ImPxl(IM,X,Y,W)     *((unsigned char*)IM + (X) + (Y)*W)

void ImCpu::InterpolateBilinear(unsigned short new_width, unsigned short new_height)
{
	unsigned short     X, Y;
	void				*new_pxl;

	/* Compute scaling factor for each dimension */
	float HeightScaleFactor = ((float)height / (float)new_height);
	float WidthScaleFactor = ((float)width / (float)new_width);

	double *ax, *ay;
	float *xd, *yd;
	unsigned short *Ix1, *Ix2, *Iy1, *Iy2;

	/* Allocate memory for the pixels */
	if (8 == bpp)
	{
		new_pxl = new char[sizeof(char) * new_width *new_height *dimension];
		xd = new float[new_width];
		yd = new float[new_height];
		Ix1 = new unsigned short[new_width];
		Ix2 = new unsigned short[new_width];
		Iy1 = new unsigned short[new_height];
		Iy2 = new unsigned short[new_height];
		
		ax = new double[new_width];
		ay = new double[new_height];
	}
	else if (16 == bpp)
	{
		new_pxl = new unsigned short[sizeof(unsigned short) * new_width *new_height *dimension];
	}

	for (Y = 0; Y < new_height; Y++)
	{
		float ydest = (float)(Y + .5)*HeightScaleFactor;
		yd[Y] = ydest;

		if (ydest <= 0.5){
			ay[Y] = 1;
			Iy1[Y] = 0;
			Iy2[Y] = 0;
		}

		if ((ydest > 0.5) && (ydest < (height - 1 + 0.5)))
		{
			/* Compute Alpha x value used to perform interpolation */
			unsigned short Integer = (unsigned short)(ydest - 0.5);
			ay[Y] = (float)((ydest - 0.5) - Integer);
			Iy1[Y] = Integer;
			Iy2[Y] = Iy1[Y] + 1;
		}

		if (ydest >= (height - 1 + 0.5))
		{
			ay[Y] = 0;
			Iy1[Y] = height - 1;
			Iy2[Y] = height - 1;
		}
	}

	for (X = 0; X < new_width; X++)
	{
		float xdest = (float)(X + .5)*WidthScaleFactor;
		xd[X] = xdest;

		if (xdest <= 0.5){
			ax[X] = 1;
			Ix1[X] = 0;
			Ix2[X] = 0;
		}

		if ( (xdest > 0.5) && (xdest < (width - 1 + 0.5)))
		{
			/* Compute Alpha x value used to perform interpolation */
     		unsigned short Integer = (unsigned short)(xdest - 0.5);
			ax[X] = (float)((xdest - 0.5) - Integer);
			Ix1[X] = Integer;
			Ix2[X] = Ix1[X] + 1;
		}

		if (xdest >= (width - 1 + 0.5))
		{
			ax[X] = 0;
			Ix1[X] = width - 1;
			Ix2[X] = width - 1;
		}
	}

	/* Compute pixel intensity in destination image */
	for (Y = 0; Y < new_height; Y++)
	{
		for (X = 0; X < new_width; X++)
		{
			/* Perform bilinear interpolation */
			ImPxl(new_pxl, X, Y, new_width) = (unsigned char)((1 - ax[X])*(1 - ay[Y])* ImPxl(pxl, Ix1[X], Iy1[Y], width) + \
																   ax[X] *(1 - ay[Y])* ImPxl(pxl, Ix2[X], Iy1[Y], width) + \
																   (1 - ax[X])*ay[Y] * ImPxl(pxl, Ix1[X], Iy2[Y], width) + \
      															       ay[Y] * ax[X] * ImPxl(pxl, Ix2[X], Iy2[Y], width));
		}
	}

	delete(pxl);
	delete(xd);
	delete(yd);

	delete(Ix1);
	delete(Ix2);
	delete(Iy1);
	delete(Iy2);

	delete(ax);
	delete(ay);

	pxl = new_pxl;

	width = new_width;
	height = new_height;

	return;
}