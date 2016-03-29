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
    by Hung-Yi Pu, Kiyun Yun, Ziri Younsi, and Suk-Jin Yoon (2016 ApJ 820,105) 
	
***********************************************************************************/
#include "Odyssey_def.h"

// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- 
//
// functions for ray-tracing																								  
//
// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- 

__device__ static void geodesic(double* Variables, double* VariablesIn, double *y, double *dydx)
{

	double r, theta, pr, ptheta;

	r		= y[0];
	theta	= y[1];
	pr		= y[4];
	ptheta	= y[5];

	
	double r2	= r * r;
	double twor = 2.0 * r;

	double sintheta = sin(theta);
	double costheta = cos(theta);
	double cos2     = costheta * costheta;
	double sin2     = sintheta * sintheta;
	double sigma	= r2 + a2 * cos2;
	double delta	= r2 - twor + a2;
	double sd		= sigma * delta;
	double siginv	= 1.0 / sigma;
	double bot		= 1.0 / sd;

	
	//avoid problems near the axis
	if (sintheta < 1e-8)
	{
		sintheta = 1e-8;
		sin2 = 1e-16;
	}

	dydx[0] = -pr * delta * siginv;	
	dydx[1] = -ptheta * siginv;
	dydx[2] = -(twor * A + (sigma - twor) * L / sin2) * bot;
	dydx[3] = -(1.0 + (twor * (r2 + a2) - twor * A * L) * bot);
	dydx[4] = -(((r - 1.0) * (-kappa) + twor * (r2 + a2) - 2.0 * A * L) * bot - 2.0 * pr * pr * (r - 1.0) * siginv);
	dydx[5] = -sintheta * costheta*(L * L / (sin2 * sin2) - a2) * siginv;
}



__device__ static void rkstep(double* Variables, double* VariablesIn,double *y, double *dydx, double h, double *yout, double *yerr)
{
	int i;
	double ak[N];
	double ytemp1[N], ytemp2[N], ytemp3[N], ytemp4[N], ytemp5[N];
	double hdydx;
	double yi, yt;

	for (i = 0; i < N; i++)
	{
		hdydx     = h * dydx[i];
		yi        = y[i];
		ytemp1[i] = yi + 0.2 * hdydx;
		ytemp2[i] = yi + (3.0/40.0) * hdydx;
		ytemp3[i] = yi + 0.3 * hdydx;
		ytemp4[i] = yi -(11.0/54.0) * hdydx;
		ytemp5[i] = yi + (1631.0/55296.0) * hdydx;
		yout[i]   = yi + (37.0/378.0) * hdydx;
		yerr[i]   = ((37.0/378.0)-(2825.0/27648.0)) * hdydx;
	}

	geodesic(Variables, VariablesIn, ytemp1, ak);

	for (i = 0; i < N; i++)
	{
		yt         = h * ak[i];
		ytemp2[i] += (9.0/40.0) * yt;
		ytemp3[i] -= 0.9 * yt;
		ytemp4[i] += 2.5 * yt;
		ytemp5[i] += (175.0/512.0) * yt;
	}

	geodesic(Variables, VariablesIn, ytemp2, ak);

	for (i = 0; i < N; i++)
	{
		yt         = h * ak[i];
		ytemp3[i] += 1.2 * yt;
		ytemp4[i] -= (70.0/27.0) * yt;
		ytemp5[i] += (575.0/13824.0) * yt;
		yout[i]   += (250.0/621.0) * yt;
		yerr[i]   += ((250.0/621.0)-(18575.0/48384.0)) * yt;
	}

		geodesic(Variables, VariablesIn, ytemp3, ak);

	for (i = 0; i < N; i++)
	{
		yt         = h * ak[i];
		ytemp4[i] += (35.0/27.0) * yt;
		ytemp5[i] += (44275.0/110592.0) * yt;
		yout[i]   += (125.0/594.0) * yt;
		yerr[i]   += ((125.0/594.0)-(13525.0/55296.0)) * yt;
	}

	geodesic(Variables, VariablesIn, ytemp4, ak);

	for (i = 0; i < N; i++)
	{
		yt         = h * ak[i];
		ytemp5[i] += (253.0/4096.0) * yt;
		yerr[i]   -= (277.0/14336.0) * yt;
	}

	geodesic(Variables, VariablesIn, ytemp5, ak);

	for (i = 0; i < N; i++)
	{
		yt       = h * ak[i];
		yout[i] += (512.0/1771.0) * yt;
		yerr[i] += ((512.0/1771.0)-0.25) * yt;
	}
}


__device__ static double rk5(double* Variables, double* VariablesIn, double *y, double *dydx, double htry, double escal, double *yscal, double *hdid)
{



	int i;

	double hnext;
	double errmax, h = htry, htemp;
	double yerr[N], ytemp[N];

	while (1)
	{
		// find adaptive step size
		rkstep(Variables, VariablesIn, y, dydx, h, ytemp, yerr);

		errmax = 0.0;
		for (i = 0; i < N; i++)
		{
			double temp = fabs(yerr[i]/yscal[i]);
			if (temp > errmax) errmax = temp;
		}

		errmax *= escal;
		if (errmax <= 1.0) break;

		htemp = 0.9 * h / sqrt(sqrt(errmax));

		h *= 0.1;

		if (h >= 0.0)
		{
			if (htemp > h) h = htemp;
		}
		else
		{
			if (htemp < h) h = htemp;
		}
	}

	if (errmax > 1.89e-4)
	{
		hnext = 0.9 * h * pow(errmax, -0.2);
	}
	else
	{
		hnext = 5.0 * h;
	}

	*hdid = h;

	memcpy(y, ytemp, N * sizeof(double));

	return hnext;
}

    
__device__ static void initial(double* Variables, double* VariablesIn, double *y0, double *ydot0)
{
    double alpha = grid_x;
    double beta  = grid_y;
    double x     = sqrt(r0*r0+a2)*sin(theta0)-beta*cos(theta0);
    double y     = alpha;
    double z     = r0*cos(theta0)+beta*sin(theta0);
    double u     = x*x+y*y+z*z-a2;
    
    y0[0] = sqrt((u+sqrt(u*u+(2.*A*z)*(2.*A*z)))/2.);   
    y0[1] = acos(z/y0[0]);                              
    y0[2] = atan2(y,x);                                
    y0[3] = 0;
        
        
    double r = y0[0];
    double theta = y0[1];
    double phi=y0[2];
        
    double sigma = r*r+(A*cos(theta))*(A*cos(theta));
    u     = sqrt(a2+r*r);
    double v     = -sin(theta0)*cos(phi);
    double zdot=-1.;
        
    double rdot0 = zdot*(-u*u*cos(theta0)*cos(theta)+r*u*v*sin(theta))/sigma;         
    double thetadot0 = zdot*(cos(theta0)*r*sin(theta)+u*v*cos(theta))/sigma;          
    double phidot0 = zdot*sin(theta0)*sin(phi)/(u*sin(theta));                         
    
    ydot0[0] = rdot0;
    ydot0[1] = thetadot0;
    ydot0[2] = phidot0;
 
        
    double sintheta, costheta;
    sintheta=sin(theta);
    costheta=cos(theta);
    double cos2 = costheta*costheta;
    double sin2 = sintheta*sintheta;
    
    
    double r2 = r * r;
    double delta = r2 - 2.0 * r + a2;
    double s1 = sigma - 2.0 * r;
    
    y0[4]= rdot0*sigma/delta;
    y0[5]= thetadot0*sigma;
    
    
    double energy2 = s1*(rdot0*rdot0/delta+thetadot0*thetadot0)
    				+ delta*sin2*phidot0*phidot0;
    
    double energy = sqrt(energy2);
    
    // rescaled by energy
    y0[4] = y0[4]/energy;
    y0[5] = y0[5]/energy;
    
    // set E = 1 
    L = ((sigma*delta*phidot0-2.0*A*r*energy)*sin2/s1)/energy;
    kappa = y0[5]*y0[5]+a2*sin2+L*L/sin2;

    
}


__device__ static float ISCO(double* VariablesIn)
{
	double z1		   = 1 + pow(1 - A * A, 1 / 3.0) * pow(1 + A, 1 / 3.0) + pow(1 - A, 1 / 3.0);
	double z2		   = sqrt(3 * A * A + z1 * z1);
	return 3. + z2 - sqrt((3 - z1) * (3 + z1 + 2 * z2));
}

// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- 
//
// functions for radiative transfer																								  
//
// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- 


#define Te_min    0.1
#define Te_max    100.
#define Te_grids  50.

static __device__ __constant__ double K2_tab[] = {
-10.747001,
-9.362569,
-8.141373,
-7.061568,
-6.104060,
-5.252153,
-4.491244,
-3.808555,
-3.192909,
-2.634534,
-2.124893,
-1.656543,
-1.223007,
-0.818668,
-0.438676,
-0.078863,
+0.264332,
+0.593930,
+0.912476,
+1.222098,
+1.524560,
+1.821311,
+2.113537,
+2.402193,
+2.688050,
+2.971721,
+3.253692,
+3.534347,
+3.813984,
+4.092839,
+4.371092,
+4.648884,
+4.926323,
+5.203493,
+5.480457,
+5.757264,
+6.033952,
+6.310550,
+6.587078,
+6.863554,
+7.139990,
+7.416395,
+7.692778,
+7.969143,
+8.245495,
+8.521837,
+8.798171,
+9.074500,
+9.350824,
+9.627144
};


__device__ static double K2_find(double Te)
{
  double d = Te_grids*(log(Te / Te_min)/ log(Te_max / Te_min));
  int    i = floor(d);

  return (1 - (double)(d-i)) * K2_tab[i] + (double)(d-i) * K2_tab[i+1];
}

__device__ static  double K2(double Te)
{
  double tab_K2;

	if (Te>Te_max){ 
		tab_K2=2.*Te*Te;
		return exp(tab_K2);
	}
	  
    tab_K2= K2_find(Te);
        return exp(tab_K2);
}



__device__ static double Jansky_Correction(double* VariablesIn,double ima_width)
{
	double distance=C_sgrA_d*C_pc;
	double theta=atan(ima_width*C_sgrA_mbh*C_rgeo/distance);
	double pix_str=theta/(SIZE/2.)*theta/(SIZE/2.);  //Size of image in steradians
	return pix_str/C_Jansky;
}

__device__ static double Luminosity_Correction(double* VariablesIn,double ima_width)
{
        double distance=C_sgrA_d*C_pc;
	double theta=atan(ima_width*C_sgrA_mbh*C_rgeo/distance);
	double pix_str=theta/(SIZE/2.)*theta/(SIZE/2.);  //Size of image in steradians
	return pix_str*distance*distance*4.*PI;
}

