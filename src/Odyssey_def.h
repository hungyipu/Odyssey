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
#include <device_launch_parameters.h>
#include <iostream>
#include <string.H>
#include <math.h>
#include <stdio.h>


#define N					  6
#define PI					  3.14159265
#define C_G					  6.67259e-08
#define C_c					  2.99792458e+10
#define C_mSun					  1.99e+33
#define C_rgeo					  1.4774e+05        //C_G*C_msun/C_c/C_c
#define C_h					  6.6260755e-27	    //PlanCk Constant
#define C_kB					  1.380658e-16	    //Boltzmann constant
#define C_e					  4.8032068e-10
#define C_me					  9.1093897e-28
#define C_mp					  1.6726231e-24
#define C_Jansky				  1e-23
#define C_ly					  9.463e17
#define C_pc					  3.086e18
#define C_sgrA_mbh				  4.3e6             //mass of the black hole (Sgr A*)
#define C_sgrA_d				  8500              //distance to  the black hole (Sgr A*), in unit of pc
								  
								  
								  
								  
#define VarNUM					  9
#define r0					  Variables[0]
#define theta0  				  Variables[1]
#define a2   					  Variables[2]
#define Rhor    				  Variables[3]
#define Rmstable    			  	  Variables[4]
#define L   					  Variables[5]
#define kappa    				  Variables[6]
#define grid_x   				  Variables[7]
#define grid_y   				  Variables[8]
								  
// For Local Memory				  
#define VarINNUM				  4
#define A					  VariablesIn[0]
#define INCLINATION				  VariablesIn[1]
#define SIZE					  VariablesIn[2]
#define freq_obs				  VariablesIn[3]

// Mapping from threadIdx/blockIdx to pixel position
#define ResultsPixel(q)    	                  ResultsPixel[3 * ((int)SIZE * (Y1) + (X1)) + q]
#define X1					  GridIdxX * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x
#define Y1					  GridIdxY * gridDim.y * blockDim.y + blockDim.y * blockIdx.y + threadIdx.y
