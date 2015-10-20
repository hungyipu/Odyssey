# Odyssey
Odyssey is a public, GPU-based General Relativistic Radiative Transfer (GRRT) code for computing images and/or spectra in Kerr metric, which described the spacetime aroung a rotating black hole. Implemented in CUDA C/C++, Odyssey is based on the ray-tracing algorithm presented in [Fuerst & Wu (2004)](http://adsabs.harvard.edu/abs/2004A%26A...424..733F), and radiative transfer formulation described in [Younsi et al. (2012)](http://adsabs.harvard.edu/abs/2012A%26A...545A..13Y).

For flexibility, namespace structure in C++  is used for different tasks. Two defalut tasks are presented in the sourse code. Including :

 1. The redshift of the Keplerian disk</li>
 2. The image of Keplerian rotating shell at 340GHz</li>
 ([here](https://github.com/hungyipu/Odyssey/wiki/Default-Tasks-of-Odyssey-Source-Code) shows the computed results)
  
 
## Summary of Source Codes
Odyssey source code prvided in the src folder includes the following files:<br />
<br />
**main.cpp**
assign parameters (black hole spin, inclinaiton angle, image size, observed frequency, CUDA configuration...), call Task, then save CUDA computed result to output file<br />

**task1.h**
declares namespace for Task1<br />
 
 **task1.cpp**
defines functions for setting up CUDA computation for Task1, including `setDims()`, `PRE()`, `GPUcompute()`, and `AFTER()`<br />

**task2.h**
 declares namespace for Task2

 
**task2.cpp**
defines functions for setting up CUDA computation for Task2<br />

 
**Odyssey.cu**
 describes jobs of specific Task, such as `__global__ GPU_task1work()`, `__global__ GPU_task1work()`. Computation result will retun to **main.cpp**<br />
 
**Odyssey_def.h**
 defines constants (such as black hole mass, distance to the black hole),   
 and variables which will be saved in the GPU global memory during computation<br />
 
 
**Odyssey_def_fun.h**
 defines functions needed for:
 <ul>
 <li>Ray-Tracing</li>
 such as initial condition  `initial()` , diffrential equaitons for geodesics, adaptive size Runge-Kutta method `rk5()`
 <li>Radiative Transfer</li>
 such as table of Bessel function of the second kind (for computation of thermal synchoron emission), unit conversion to Jansky or Liminosity (erg/sec)
 </ul>


## Code Structure
The flow chart for the code structure of Odyssey is provided below.

In **main.cpp**, `task1()` is called by `main()`, then go through<br />
<br />
`task1()`:<br />
|---assign parameters <br />
|<br />
|---set CUDA configuration `setDims()`<br />
|<br />
|---allocate memory on device for input and output `PRE()`<br />
|<br />
|---perform the [*for-loop* for GRRT](https://github.com/hungyipu/Odyssey/wiki/How-Odyssey-Works) `GPUcompute()`<br />
|<br />
|---copy memory form device to host and free CUDA memory `AFTER()`<br />
|<br />
|---save result to ouput<br />


## Code Structure: more details
By calling `GPUcompute()`, the parallel computation will be performed according to the job-detials described inside `__global__ GPU_task1work()` in **Odyssey.cu**.<br />
<br />
`__global__ GPU_task1work()`:<br />
|---setup initial condition `initial()` <br />
|<br />
|================Loop Start=====================<br />
|--- update the ray backward in time by adaptive size, Runge-Kutta method `rk5()`<br />
```
job-details:
   ex. when the ray hit the disk, compute the redshift
       (you can define a different job here)
```
|--- exit if the ray enter the black hole or outside the region of interest, otherwise, contine the Loop<br />
|================Loop End=====================<br />

<br />
Odyssey is fast, accurate, and flexible. User can easiliy assign a different job by simply modifying the job-details. 
<br />
<br />Alternatively, User can also add a new Task (e.g., task 3) by following suggested recipe:
 1. add file: task3.h</li>
 2. add file: task3.cpp</li>
 3. add subroutine: task3() in **main.cpp**
 4. add related subroutines and describe job-details in **Odyssey.cu**
 
## Reference
"Odyssey: A PUBLIC GPU-BASED CODE FOR GENERAL-RELATIVISTIC RADIATIVE TRANSFER IN KERR
SPACE-TIME" by Hung-Yi Pu, Kiyun Yun, Ziri Younsi, and Suk-Jin Yoon

<img align="center" src="http://www.asiaa.sinica.edu.tw/~hypu/logo_asiaa.png" width="60">
<img align="center" src="http://www.asiaa.sinica.edu.tw/~hypu/logo_yonsei.jpg" width="65">
<img align="center" src="http://www.asiaa.sinica.edu.tw/~hypu/logo_GU.gif" width="120">

## Odyssey_Edu
An educational software, [Odyssey_Edu] (https://odysseyedu.wordpress.com/), is devloped together with Odyssey for visualizing the ray trajectories in Kerr spacetime.


