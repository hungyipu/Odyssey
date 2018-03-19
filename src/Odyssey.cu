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

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include "Odyssey_def.h"
#include "Odyssey_def_fun.h"

	
// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- 
//
// task1: 																										  
//
// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- 


__device__ double task1fun_GetZ(double* Variables, double* VariablesIn, double *y)
{
	double r1 = y[0];
	double E_local= -(r1 * r1 + A * sqrt(r1)) / (r1 * sqrt(r1 * r1 - 3. * r1 + 2. * A * sqrt(r1))) + L / sqrt(r1) / sqrt(r1 * r1 - 3. * r1  +2. * A * sqrt(r1));
	double E_inf= -1.0;      
	return E_local/E_inf; 
}




__global__ void GPU_task1work(double* ResultsPixel, double* VariablesIn, int GridIdxX, int GridIdxY)
{
	// to check whether the photon is inside image plane
	if(X1 >= SIZE || Y1 >= SIZE) return;  


	
	double Variables[VarNUM];

	


	r0       = 1000.0;						     
	theta0   = (PI/180.0) * INCLINATION;		 
	a2       = A * A;						    
	Rhor     = 1.0 + sqrt(1.0 - a2) + 1e-5;      
	Rmstable = ISCO(VariablesIn);				        
	

	double htry = 0.5, escal = 1e14, hdid = 0.0, hnext = 0.0;
	double y[N], dydx[N], yscal[N], ylaststep[N];

	double Rdisk	 = 50.;
	double ima_width = 55.;
        double offset_shoot = 2.*ima_width/(int)SIZE; 
        
        double s1  = ima_width;                      
        double s2  = 2.*ima_width/((int)SIZE+1.);   
       
        grid_x = -s1 + s2*(X1+1.);
        grid_y = -s1 + s2*(Y1+1.);
        

	initial(Variables, VariablesIn, y, dydx);


	ResultsPixel(0) = grid_x;
	ResultsPixel(1) = grid_y;
	ResultsPixel(2) = 0;
	

	while (1)
	{
		for(int i = 0; i < N; i++)
			ylaststep[i] = y[i];
		
		geodesic(Variables, VariablesIn, y, dydx);




		for (int i = 0; i < N; i++)
			yscal[i] = fabs(y[i]) + fabs(dydx[i] * htry) + 1.0e-3;

		//fifth-order Runge-Kutta method
		hnext = rk5(Variables, VariablesIn, y, dydx, htry, escal, yscal, &hdid);

		    
		// hit the disk, compute redshift
		if( y[0] < Rdisk && y[0] > Rmstable && (ylaststep[1] - PI/2.) * (y[1] - PI/2.) < 0. )
		{		
			ResultsPixel(2) = 1./task1fun_GetZ(Variables, VariablesIn, y);
			break;
		}

		// Inside the event horizon radius or escape to infinity
		if ((y[0] > r0) && (dydx[0]>0)) break;
      
		if (y[0] < Rhor) break;



		htry = hnext;
	}

}


extern "C"
void GPU_assigntask1(double* ResultsPixel, double* VariablesIn, int GridIdxX, int GridIdxY,
						int GridDimX, int GridDimY, int BlockDimX, int BlockDimY)
{
	dim3 GridDim, BlockDim;
	
	GridDim.x  = GridDimX;
	GridDim.y  = GridDimY;
	BlockDim.x = BlockDimX;
	BlockDim.y = BlockDimY;
	
		
	GPU_task1work<<<GridDim, BlockDim>>>(ResultsPixel, VariablesIn, GridIdxX, GridIdxY);
	cudaThreadSynchronize();
}
// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- 
//
// end of task1:																									  
//
// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- 




// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- 
//
// task2: 
//
// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- 
__device__ double task2fun_GetZ(double* Variables, double* VariablesIn, double *y)
{
    double ut,uphi,ur,E_local;
    double E_inf= -1.0;   
    double r=y[0];
    double theta=y[1];
    double pr=y[4];
        
    double r2 = r*r;
    double twor = 2.0*r;
    double sintheta, costheta;
    sintheta=sin(theta);
    costheta=cos(theta);
    double cos2 = costheta*costheta;
    double sin2 = sintheta*sintheta;
   
   
   
    double sigma = r2+a2*cos2;
    double delta = r2-twor+a2;
    double bigA=(r * r + A * A) * (r * r + A * A) - A * A * delta * sin2;
   
    if( r<Rmstable)  
     {
     
       double lambda=(Rmstable*Rmstable-2.*A*sqrt(Rmstable)+a2)/(sqrt(Rmstable*Rmstable*Rmstable)-2.*sqrt(Rmstable)+A);
       double gamma=sqrt(1-2./3./Rmstable);
       double h=(2.*r-A*lambda)/delta;
           
       
        
       ur=-sqrt(2./3./Rmstable)*pow((Rmstable/r-1.),1.5);
       uphi=gamma/r/r*(lambda+A*h);
       ut=gamma*(1.+2/r*(1.+h));
     
       E_local=-ut+L*uphi+pr*ur;
       return E_local/E_inf; 
    }
   
    E_local=-(r*r+A*sqrt(r))/(r*sqrt(r*r-3.*r+2.*A*sqrt(r)))+L/sqrt(r)/sqrt(r*r-3.*r+2.*A*sqrt(r)); 
       
	return E_local/E_inf; 
}




__global__ void GPU_task2work(double* ResultsPixel, double* VariablesIn, int GridIdxX, int GridIdxY)
{

	if(X1 >= SIZE || Y1 >= SIZE) return;  

	double Variables[VarNUM];
	r0       = 1000.0;						     
	theta0   = (PI/180.0) * INCLINATION;		 
	a2       = A * A;						    
	Rhor     = 1.0 + sqrt(1.0 - a2) + 1e-5;   
	Rmstable = ISCO(VariablesIn);	
	

	double htry = 0.5, escal = 1e14, hdid = 0.0, hnext = 0.0;
	double y[N], dydx[N], yscal[N], ylaststep[N];
	
	double Rdisk	 = 500.;
	double ima_width = 10.;
        
    double s1  = ima_width;                      
    double s2  = 2.*ima_width/((int)SIZE+1.);   
        

	double Jy_corr=Jansky_Correction(VariablesIn,ima_width);
    double L_corr=Luminosity_Correction(VariablesIn,ima_width);


	          
    grid_x = -s1 + s2*(X1+1.);
    grid_y = -s1 + s2*(Y1+1.);
           
    
    initial(Variables, VariablesIn, y, dydx);

    ResultsPixel(0) = grid_x;
	ResultsPixel(1) = grid_y;
	ResultsPixel(2) = 0;
	

	double ds=0.;	
	double dtau=0.;
	double dI=0.;
	
		
	while (1)
	{
			for(int i = 0; i < N; i++)
				ylaststep[i] = y[i];
		
			geodesic(Variables, VariablesIn, y, dydx);

			for (int i = 0; i < N; i++)
				yscal[i] = fabs(y[i]) + fabs(dydx[i] * htry) + 1.0e-3;

			hnext = rk5(Variables, VariablesIn, y, dydx, htry, escal, yscal, &hdid);


			if ((y[0] > r0) && (dydx[0]>0)){
				ResultsPixel(2) = dI*freq_obs*freq_obs*freq_obs*L_corr; 
				break;
      		}

			if (y[0] < Rhor){
				ResultsPixel(2) = dI*freq_obs*freq_obs*freq_obs*L_corr; 
				break;
			}


			double intensity,flux,luminosity;
			double local_emi=0.;
			double local_abs=0.;
			double r=y[0];
			double theta=y[1];


			if(y[0]<Rdisk){

				double zzz        = task2fun_GetZ(Variables, VariablesIn, y); //zzz=E_em/E_obs
				double freq_local = freq_obs*zzz;	

				//****************************************************
				//****************** thermal synchrotron**************
				//****************************************************
				double nth0=3e7;
				double zc=r*cos(theta);
				double rc=r*sin(theta);

				double nth=nth0*exp(-zc*zc/2./rc/rc)*pow(r,-1.1);
				double Te=1.7e11*pow(r,-0.84); 
				double b=sqrt(8.*PI*0.1*nth*C_mp*C_c*C_c/6./r);
 
				double vb=C_e*b/2./PI/C_me/C_c;
				double theta_E= C_kB*Te/C_me/C_c/C_c;
				double v=freq_local;
				double x=2.*v/3./vb/theta_E/theta_E;
				
				double K_value=K2(theta_E);

				double comp1=4.*PI*nth*C_e*C_e*v/sqrt(3.)/K_value/C_c;
				double comp2=4.0505/pow(x,(1./6.))*(1.+0.4/pow(x,0.25)+0.5316/sqrt(x))*exp(-1.8899*pow(x,1./3.));
				double j_nu=comp1*comp2;
				double B_nu=2.0*v*v*v*C_h/C_c/C_c/(exp(C_h*v/C_kB/Te)-1.0);


				//****************************************************
				//******************integrate intensity along ray ****
				//****************************************************
				ds		=  htry;

				dtau	=  dtau   + ds*C_sgrA_mbh*C_rgeo*j_nu/B_nu*zzz;  
				dI		=  dI	  + ds*C_sgrA_mbh*C_rgeo*j_nu/freq_local/freq_local/freq_local*exp(-dtau)*zzz;  
		
				
			}
	

		htry = hnext;
	}

}


extern "C"
void GPU_assigntask2(double* ResultsPixel, double* VariablesIn, int GridIdxX, int GridIdxY,
						int GridDimX, int GridDimY, int BlockDimX, int BlockDimY)
{
	dim3 GridDim, BlockDim;
	
	GridDim.x  = GridDimX;
	GridDim.y  = GridDimY;
	BlockDim.x = BlockDimX;
	BlockDim.y = BlockDimY;
	
		
	GPU_task2work<<<GridDim, BlockDim>>>(ResultsPixel, VariablesIn, GridIdxX, GridIdxY);
	cudaThreadSynchronize();
}
// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- 
//
// end of task2																								  
//
// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- 





// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- 
//
// task3: your turn...
//
// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- 

