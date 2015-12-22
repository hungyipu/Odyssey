/***********************************************************************************
    Copyright 2015  Hung-Yi Pu, Kiyun Yun, Ziri Yonsi, Sunk-Jin Yoon
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
    by Hung-Yi Pu, Kiyun Yun, Ziri Younsi, and Suk-Jin Yoon (submitted to ApJ) 
	
***********************************************************************************/
#include <cuda.h>
#include <cuda_runtime.h>
#include "Odyssey_def.h"

namespace Task1
{

	class mission1
	{
	private:
	

	    double*  d_ResultsPixel;
	    double*  d_VariablesIn;
	    
	    int      mGridDimx;
	    int      mGridDimy;
	    int      mBlockDimx;
	    int      mBlockDimy;
	    int	     mSize;


	public:
		void setDims(int GridDimX, int GridDimY, int BlockDimX, int BlockDimY);
		void PRE(double* VariablesIn);
		void GPUCompute(int GridIdxX, int GridIdxY);
		void AFTER(double* ResultHit);

	};

}


