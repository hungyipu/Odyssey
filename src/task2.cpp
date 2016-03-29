/***********************************************************************************
    Copyright 2015  Hung-Yi Pu, Kiyun Yun, Ziri Younsi, Sunk-Jin Yoon
                        Odyssey  version 1.0   (released  2015)
    This file is part of Odyssey source code. Odyssey is a public, GPU-based code 
    for General Relativistic Radiative Transfer (GRRT), following the 
    ray-tracing algorithm presented in 
    Fuerst, S. V., & Wu, K. 2007, A&A, 474, 55, 
    and the radiative transfer formulation described in 
    Younsi, Z., Wu, K., & Fuerst, S. V. 2012, A&A, 545, A13
    
    Odyssey is distributed freely under the GNU general public license. 
    You can redistribute it and/or modify it under the terms of the License
        http://www.gnu.org/licenses/gpl.txt
    The current distribution website is:
	https://github.com/hungyipu/Odyssey/ 
	
    We ask that users of Odyssey cite the following paper in their subsequent scientific 
    literature and publications which result from the use of any part of Odyssey:
    "Odyssey: A PUBLIC GPU-BASED CODE FOR GENERAL-RELATIVISTIC RADIATIVE TRANSFER 
    IN KERR SPACE-TIME" 
    by Hung-Yi Pu, Kiyun Yun, Ziri Younsi, and Suk-Jin Yoon (2016 ApJ 820, 105) 
	
***********************************************************************************/
#include "task2.h"

namespace Task2
{

	void mission2::setDims(int GridDimX, int GridDimY, int BlockDimX, int BlockDimY)
	{
		mGridDimx  = GridDimX;
		mGridDimy  = GridDimY;
		mBlockDimx = BlockDimX;
		mBlockDimy = BlockDimY;
	}
	

	void mission2::PRE(double* VariablesIn)
	{

		mSize = (int)SIZE;

	    cudaMalloc(&d_ResultsPixel           , sizeof(double) * mSize * mSize * 3);
		cudaMalloc(&d_VariablesIn        , sizeof(double) * VarINNUM);
		cudaMemcpy(d_VariablesIn	 , VariablesIn	 , sizeof(double) * VarINNUM	, cudaMemcpyHostToDevice);
	}


	
	void mission2::AFTER(double* ResultHit)
	{
		cudaMemcpy(ResultHit, d_ResultsPixel, sizeof(double) * mSize * mSize * 3, cudaMemcpyDeviceToHost);
	
		cudaFree(d_ResultsPixel);
		cudaFree(d_VariablesIn);
	}



	extern "C" 
	void GPU_assigntask2(double* ResultsPixel, double* VariablesIn, int GridIdxX, int GridIdxY,
									int GridDimX, int GridDimY, int BlockDimX, int BlockDimY);

	void mission2::GPUCompute(int GridIdxX, int GridIdxY)
	{
		GPU_assigntask2(d_ResultsPixel, d_VariablesIn, GridIdxX, GridIdxY,
						mGridDimx, mGridDimy, mBlockDimx, mBlockDimy);
	}

}
